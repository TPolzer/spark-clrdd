import org.jocl.CL._
import org.jocl._
import org.slf4j.LoggerFactory
import scala.collection.mutable.ArrayBuffer
import java.util.HashMap
import java.lang.ref.{ReferenceQueue, WeakReference}
import java.util.concurrent.{Future, CompletableFuture}
import java.util.concurrent.atomic.AtomicLong
import java.nio.ByteOrder
import java.nio.ByteBuffer

object OpenCL
  extends java.lang.ThreadLocal[OpenCLSession] // supplies one OpenCLSession per thread, round-robin over devices
{
  setExceptionsEnabled(true)
  val deviceType = CL_DEVICE_TYPE_GPU
  val devices = { // holds a function for creating a session for each device
    val numPlatforms = Array(0)
    clGetPlatformIDs(0, null, numPlatforms)
    val platforms = new Array[cl_platform_id](numPlatforms(0))
    clGetPlatformIDs(platforms.length, platforms, null)
    platforms.flatMap(platform => {
      val contextProperties = new cl_context_properties
      contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform)
      val numDevices = Array(0)
      clGetDeviceIDs(platform, deviceType, 0, null, numDevices)
      val devices = new Array[cl_device_id](numDevices(0))
      clGetDeviceIDs(platform, deviceType, numDevices(0), devices, null)
      devices.map(device => () => {
        val context = clCreateContext(contextProperties, 1, Array(device), null, null, null)
        val queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, null) //deprecated for OpenCL 2.0, needed for 1.2!
        new OpenCLSession(context, queue)
      })
    })
  }

  override protected def initialValue = {
    val threadId = java.lang.Thread.currentThread.getId
    devices((threadId % devices.size.toLong).toInt)()
  }

  /**
   * A finalizing holder for cl_program. `program` will be released upon finalization!
   */
  case class Program (program: cl_program) {
    override def finalize : Unit = {
      LoggerFactory.getLogger(getClass).info("automatically releasing " + program)
      clReleaseProgram(program)
    }
  }

  case class KernelArg (value: Pointer, size: Long)
  object KernelArg {
    def apply(mem: cl_mem) : KernelArg = KernelArg(Pointer.to(mem), Sizeof.cl_mem)
    def apply(i: Int) : KernelArg = KernelArg(Pointer.to(Array(i)), Sizeof.cl_int)
  }

  case class Dimensions (
    dim: Int,
    global_work_offset: Array[Long],
    global_work_size: Array[Long],
    local_work_size: Array[Long]
  ) {
  }

  case class Chunk (
    val size: Int, //bytes
    val handle: cl_mem,
    val ready: cl_event
  )
  extends java.io.Closeable
  {
    override def close : Unit = {
      if(handle != new cl_mem)
        clReleaseMemObject(handle)
      if(ready != new cl_event)
        clReleaseEvent(ready)
    }
  }
}

class OpenCLSession (val context: cl_context, val queue: cl_command_queue)
{
  import OpenCL.{Program, Chunk, KernelArg, Dimensions}
  val log = LoggerFactory.getLogger(getClass)
  log.info("created OpenCLSession")


  /*
   * TODO: proper cache (discard old programs)?
   */
  private val programCache = java.util.Collections.synchronizedMap(new HashMap[Seq[String], Program])
  /**
   * Build (or get from cache) an OpenCL program.
   */
  def getProgram(source: Seq[String]) : Program = {
    Option(programCache.get(source)).orElse({
        val program = clCreateProgramWithSource(context, source.size, source.toArray, null, null)
        val res = Program(program) // finalize if buildProgram throws
        clBuildProgram(res.program, 0, null, "", null, null)
        programCache.put(source, res)
        Option(res)
      }).get
  }

  var executionTime = new AtomicLong
  var queueTime = new AtomicLong

  def callKernel(program: Program, kernelName: String, args: Seq[KernelArg], dependencies: Seq[cl_event], dimensions: Dimensions) : cl_event = {
    val startTime = System.nanoTime
    val kernel = clCreateKernel(program.program, kernelName, null)
    val ready = new cl_event
    try {
      var index = 0
      val waitList = dependencies.toArray
      args.foreach({case KernelArg(value, size) => {
        clSetKernelArg(kernel, index, size, value)
        index += 1
      }})
      clEnqueueNDRangeKernel(
        queue,
        kernel,
        dimensions.dim, dimensions.global_work_offset, dimensions.global_work_size, dimensions.local_work_size,
        waitList.size,
        waitList,
        ready
      )
      clSetEventCallback(ready, CL_COMPLETE, new EventCallbackFunction() {
        override def function(event: cl_event, command_exec_callback_type: Int, user_data: AnyRef) = {
          val profiling = new Array[Long](1)
          clGetEventProfilingInfo(ready, CL_PROFILING_COMMAND_QUEUED, Sizeof.cl_ulong, Pointer.to(profiling), null)
          val queued = profiling(0)
          clGetEventProfilingInfo(ready, CL_PROFILING_COMMAND_START, Sizeof.cl_ulong, Pointer.to(profiling), null)
          val started = profiling(0)
          clGetEventProfilingInfo(ready, CL_PROFILING_COMMAND_END, Sizeof.cl_ulong, Pointer.to(profiling), null)
          val ended = profiling(0)
          log.trace("kernel took {}ms", (ended-started)/1e6)
          queueTime.addAndGet(ended - queued)
          executionTime.addAndGet(ended - started)
        }
      }, null)
      val endTime = System.nanoTime
      log.trace("enqueuing kernel took {}ms", (endTime - startTime)/1e6)
      ready
    } catch {
      case e : Throwable => {
        if(ready != new cl_event)
          clReleaseEvent(ready)
        clReleaseKernel(kernel)
        throw e
      }
    }
  }

  var ngroups = 8*1024
  var nlocal = 128

  def test(m: Int) = {
    val a = new Array[Double](1024*1024*1024/8)
    def time[A](a: => A) = { val now = System.nanoTime; val result = a; val micros = (System.nanoTime - now) / 1000; println("%f seconds".format(micros/1e6)); result};
    def f(m: Int) = {val chunk: Seq[OpenCL.Chunk] = time(OpenCL.get.stream(a.iterator.take(1024*1024*m/8),1024*1024*m)); println(time((1 to (1024*1/m)).map(i => OpenCL.get.reduceChunk(chunk(0), "0", "return x+y;")).foldLeft(0.0)({case (x,f) => x + f.get}))); chunk(0).close}
    {var a = OpenCL.get.queueTime.get; var b = OpenCL.get.executionTime.get; f(m); a = OpenCL.get.queueTime.get - a; b = OpenCL.get.executionTime.get - b; println(s"execution: ${b/1e6}ms")}
  }

  def reduceChunk(input: Chunk, identityElement: String, reduceBody: String): Future[Double] = {
    val startTime = System.nanoTime
    val program = getProgram(Array(
      "inline double f(double x, double y) {\n",
      reduceBody,
      """}
      __kernel
      __attribute__((vec_type_hint(double)))
      void reduce(__global double *input, __global double *output, __local double *scratch, int size) {
        int tid = get_local_id(0);
        int i = get_group_id(0) * get_local_size(0) + get_local_id(0);
        int gridSize = get_local_size(0) * get_num_groups(0);
        double cur = """, identityElement, """;
        for (; i<size; i = i + gridSize){
          cur = f(cur, input[i]);
        }
        scratch[tid]  = cur;
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = get_local_size(0) / 2; s>0; s = s >> 1){
          if (tid<s){
            scratch[tid]  = f(scratch[tid], scratch[(tid + s)]);
          }
          barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (tid==0){
          output[get_group_id(0)]  = scratch[0];
        }
        return;
      }"""))
    val numWorkGroups = ngroups
    val localSize = nlocal // TODO make this tunable / evaluate...?
    val globalSize = localSize * numWorkGroups
    val resBuffer : cl_mem = clCreateBuffer(context, 0, numWorkGroups * Sizeof.cl_double, null, null)
    val ready1 = callKernel(
      program, "reduce",
      KernelArg(input.handle) :: KernelArg(resBuffer) :: KernelArg(null, Sizeof.cl_double * localSize) :: KernelArg(input.size) :: Nil,
      input.ready :: Nil,
      Dimensions(1, Array(0), Array(globalSize), Array(localSize))
    )
    val ready2 = callKernel(
      program, "reduce",
      KernelArg(resBuffer) :: KernelArg(resBuffer) :: KernelArg(null, Sizeof.cl_double * localSize) :: KernelArg(numWorkGroups) :: Nil,
      ready1 :: Nil,
      Dimensions(1, Array(0), Array(localSize), Array(localSize))
    )
    val result = ByteBuffer.allocateDirect(Sizeof.cl_double)
    val future = new CompletableFuture[Double]
    val finished = new cl_event
    clEnqueueReadBuffer(queue, resBuffer, false, 0, Sizeof.cl_double, Pointer.to(result), 1, Array(ready2), finished)
    clSetEventCallback(finished, CL_COMPLETE, new EventCallbackFunction() {
      override def function(event: cl_event, command_exec_callback_type: Int, user_data: AnyRef) = {
        val profiling = new Array[Long](1)
        clGetEventProfilingInfo(finished, CL_PROFILING_COMMAND_QUEUED, Sizeof.cl_ulong, Pointer.to(profiling), null)
        val queued = profiling(0)
        clGetEventProfilingInfo(finished, CL_PROFILING_COMMAND_START, Sizeof.cl_ulong, Pointer.to(profiling), null)
        val started = profiling(0)
        clGetEventProfilingInfo(finished, CL_PROFILING_COMMAND_END, Sizeof.cl_ulong, Pointer.to(profiling), null)
        val ended = profiling(0)
        clGetEventProfilingInfo(ready1, CL_PROFILING_COMMAND_QUEUED, Sizeof.cl_ulong, Pointer.to(profiling), null)
        val stage1Queued = profiling(0)
        clGetEventProfilingInfo(ready1, CL_PROFILING_COMMAND_START, Sizeof.cl_ulong, Pointer.to(profiling), null)
        val stage1Started = profiling(0)
        log.trace("stage1 waited in queue {}ms", (stage1Started - stage1Queued)/1e6)
        clGetEventProfilingInfo(ready2, CL_PROFILING_COMMAND_END, Sizeof.cl_ulong, Pointer.to(profiling), null)
        val reduceFinished = profiling(0)
        clGetEventProfilingInfo(ready1, CL_PROFILING_COMMAND_END, Sizeof.cl_ulong, Pointer.to(profiling), null)
        val stage1Finished = profiling(0)
        log.trace("hiccup (reduce-reduce): {}ms", (reduceFinished-stage1Finished)/1e6)
        log.trace("hiccup (reduce-read): {}ms", (started-reduceFinished)/1e6)
        log.trace("read back took {}ms", (ended-started)/1e6)
//        queueTime.addAndGet(ended - queued)
//        executionTime.addAndGet(ended - started)
        clReleaseEvent(ready1) // TODO proper error handling!
        clReleaseEvent(ready2)
        clReleaseEvent(finished)
      }
    }, null)
    clSetEventCallback(finished, CL_COMPLETE, new EventCallbackFunction(){
      override def function(event: cl_event, command_exec_callback_type: Int, user_data: AnyRef): Unit = 
        future.complete(result.order(ByteOrder.nativeOrder).asDoubleBuffer.get(0))
    }, null)
    //clReleaseEvent(finished)
    //clReleaseEvent(ready1) // TODO proper error handling!
    //clReleaseEvent(ready2)
    clReleaseMemObject(resBuffer)
    val endTime = System.nanoTime
    log.trace("reduce overhead took {}ms", (endTime - startTime)/1e6)
    future
  }

  /**
   * Map one Chunk to a new one
   * @param functionBody the body of the map function 'inline double f(double x) {#functionBody}'.
   */
  def mapChunk(input: Chunk, functionBody: String) : Chunk = {
    val program = getProgram(Array(
      "inline double f(double x) {",
      functionBody,
      "}",
      "__kernel __attribute__((vec_type_hint(double)))",
      "void map(__global double *input, __global double *output) {",
        "int i = get_global_id(0);",
        "output[i] = f(input[i]);",
      "}"))
    val dimensions = Dimensions(1, Array(0), Array(input.size), null)
    val handle: cl_mem = clCreateBuffer(context, 0, input.size*Sizeof.cl_double, null, null)
    try {
      val ready = callKernel(
        program, "map",
        KernelArg(input.handle) :: KernelArg(handle) :: Nil,
        input.ready :: Nil,
        dimensions
      )
      new Chunk(input.size, handle, ready)
    } catch {
      case e : Throwable => {
        clReleaseMemObject(handle)
        throw e
      }
    }
  }

  /**
   * Put the contents of an iterator on the gpu in constant sized chunks (default 64MB size).
   * Chunks have to be closed after use
   */
  def stream(it: Iterator[Double], groupSize: Int = 1024*1024*64) : Seq[Chunk] = {
    val res = new ArrayBuffer[Chunk]
    try {
      while(it.hasNext) {
        var ready = new cl_event
        var handle : Option[cl_mem] = None
        try {
          val rawBuffer = ByteBuffer.allocateDirect(groupSize)
          val buffer = rawBuffer.order(ByteOrder.nativeOrder).asDoubleBuffer
          var copied = 0
          while(copied < groupSize/Sizeof.cl_double && it.hasNext) {
            buffer.put(copied, it.next)
            copied += 1
          }
          handle = Some(clCreateBuffer(context, 0, copied*8, null, null))
          clEnqueueWriteBuffer(queue, handle.get, false, 0, copied*8, Pointer.to(rawBuffer), 0, null, ready)
          res += Chunk(copied, handle.get, ready)
        } catch {
          case e:Throwable => {
            if(ready != new cl_event)
              clReleaseEvent(ready)
            handle.foreach(clReleaseMemObject)
            throw e
          }
        }
      }
      res
    } catch {
      case e:Throwable => {
        res.foreach(_.close)
        throw e
      }
    }
  }

  override def finalize = {
    log.info("finalizing OpenCLSession")
    clReleaseCommandQueue(queue)
    clReleaseContext(context)
  }
}
