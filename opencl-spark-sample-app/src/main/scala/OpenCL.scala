import org.jocl.CL._
import org.jocl._
import org.slf4j.LoggerFactory
import scala.collection.mutable.ArrayBuffer
import java.util.HashMap
import java.lang.ref.{ReferenceQueue, WeakReference}
import java.util.concurrent.{Future, CompletableFuture, ConcurrentHashMap}
import java.util.concurrent.atomic.AtomicLong
import java.nio.ByteOrder
import java.nio.ByteBuffer

object OpenCL
  extends java.lang.ThreadLocal[OpenCLSession] // supplies one OpenCLSession per device, round-robin over threads
{
  setExceptionsEnabled(true)
  val CPU = false
  lazy val deviceType = if(CPU) CL_DEVICE_TYPE_CPU else CL_DEVICE_TYPE_GPU
  lazy val devices = { // holds one session for each device
    val numPlatforms = Array(0)
    clGetPlatformIDs(0, null, numPlatforms)
    val platforms = new Array[cl_platform_id](numPlatforms(0))
    clGetPlatformIDs(platforms.length, platforms, null)
    platforms.flatMap(platform => {
      try {
        val contextProperties = new cl_context_properties
        contextProperties.addProperty(CL_CONTEXT_PLATFORM, platform)
        val numDevices = Array(0)
        clGetDeviceIDs(platform, deviceType, 0, null, numDevices)
        val devices = new Array[cl_device_id](numDevices(0))
        clGetDeviceIDs(platform, deviceType, numDevices(0), devices, null)
        devices.flatMap(device => {
          try{
            val context = clCreateContext(contextProperties, 1, Array(device), null, null, null)
            val queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, null) //deprecated for OpenCL 2.0, needed for 1.2!
            Some(new OpenCLSession(context, queue, device))
          } catch {
            case e: CLException => None
          }
        })
      } catch {
        case e: CLException => Nil
      }
    })
  }

  override protected def initialValue = {
    val threadId = java.lang.Thread.currentThread.getId
    devices((threadId % devices.size.toLong).toInt)
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
    val handle: cl_mem
  )
  extends java.io.Closeable
  {
    clRetainMemObject(handle)
    override def close : Unit = {
      clReleaseMemObject(handle)
    }
  }
}

class OpenCLSession (val context: cl_context, val queue: cl_command_queue, val device: cl_device_id)
{
  import OpenCL.{Program, Chunk, KernelArg, Dimensions}
  val log = LoggerFactory.getLogger(getClass)
  log.info("created OpenCLSession")


  /*
   * TODO: proper cache (discard old programs)?
   * simple strategy would be to just periodically dump all entries (for
   * CPU/GPU recompilation shouldn't be too expensive)
   */
  private val programCache = new ConcurrentHashMap[Seq[String],Program]
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

  class ProfilingCallback(ready: cl_event) extends EventCallbackFunction {
    clRetainEvent(ready)
    override def function(ready: cl_event, command_exec_callback_type: Int, user_data: AnyRef) = {
      try {
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
      } finally clReleaseEvent(ready)
    }
  }

  def callKernel(program: Program, kernelName: String, args: Seq[KernelArg], dependencies: Array[cl_event], dimensions: Dimensions, ready: cl_event) : Unit = {
    val startTime = System.nanoTime
    var kernel : Option[cl_kernel] = None
    try {
      kernel = Some(clCreateKernel(program.program, kernelName, null))
      log.trace("createKernel took {}ms", (System.nanoTime - startTime)/1e6)

      var index = 0
      val argTime = System.nanoTime
      args.foreach({case KernelArg(value, size) => {
        clSetKernelArg(kernel.get, index, size, value)
        index += 1
      }})
      log.trace("KernelArgs took {}ms", (System.nanoTime - argTime)/1e6)

      val enTime = System.nanoTime
      clEnqueueNDRangeKernel(
        queue,
        kernel.get,
        dimensions.dim, dimensions.global_work_offset, dimensions.global_work_size, dimensions.local_work_size,
        if(dependencies != null) dependencies.size else 0,
        dependencies,
        ready
      )
      log.trace("clEnqueueNDRangeKernel took {}ms", (System.nanoTime - enTime)/1e6)

      clSetEventCallback(ready, CL_COMPLETE, new ProfilingCallback(ready), null)

      val endTime = System.nanoTime
      log.trace("callKernel took {}ms", (endTime - startTime)/1e6)
    } finally {
      kernel.foreach(clReleaseKernel)
    }
  }

  /*
   * parallelization happens outside OpenCL for CPU
   */
  var ngroups = if(OpenCL.CPU) 1 else 8*1024
  var nlocal = if(OpenCL.CPU) 1 else 128
  
  var testSize = 1
  lazy val a = new Array[Double](1024*1024*1024/8)
  def test(m: Int) = {
    def time[A](a: => A) = { val now = System.nanoTime; val result = a; val micros = (System.nanoTime - now) / 1000; println("%f seconds".format(micros/1e6)); result};
    def f(m: Int) = {val chunk: Seq[OpenCL.Chunk] = time(stream(a.iterator.take(1024*1024*m/8),1024*1024*m)); clFinish(queue); println(time((1 to (1024*testSize/m)).map(i => reduceChunk(chunk(0), "0", "return x+y;")).foldLeft(0.0)({case (x,f) => x + f.get}))); chunk(0).close}
    {var a = queueTime.get; var b = executionTime.get; f(m); a = queueTime.get - a; b = executionTime.get - b; println(s"execution: ${b/1e6}ms")}
  }

  def reduceChunk(input: Chunk, identityElement: String, reduceBody: String): Future[Double] = {
    val startTime = System.nanoTime
    val program = if (OpenCL.CPU)
      getProgram(Array(
      "inline double f(double x, double y) {\n",
      reduceBody,
      """}
      __kernel
      __attribute__((vec_type_hint(double)))
      void reduce(__global const double *input, __global double *output, __local double *scratch, int size) {
        if(get_global_id(0) == 0) {
          double cur = """, identityElement, """;
          for(int i=0; i<size; ++i) {
            cur  = f(cur, input[i]);
          }
          output[0] = cur;
        } else {
          output[get_group_id(0)] = """, identityElement, """;
        }
        return;
      }"""))
    else
      getProgram(Array(
      "inline double f(double x, double y) {\n",
      reduceBody,
      """}
      __kernel
      __attribute__((vec_type_hint(double)))
      void reduce(__global const double *input, __global double *output, __local double *scratch, int size) {
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
    log.trace("getProgram took {}ms", (System.nanoTime - startTime)/1e6)
    val numWorkGroups = ngroups
    val localSize = nlocal // TODO make this tunable / evaluate...?
    val globalSize = localSize * numWorkGroups
    val resBuffer : cl_mem = clCreateBuffer(context, 0, numWorkGroups * Sizeof.cl_double, null, null)
    val finished = new cl_event
    val ready1 = new cl_event
    val ready2 = new cl_event
    try {
      callKernel(
        program, "reduce",
        KernelArg(input.handle) :: KernelArg(resBuffer) :: KernelArg(null, Sizeof.cl_double * localSize) :: KernelArg(input.size) :: Nil,
        null,
        Dimensions(1, Array(0), Array(globalSize), Array(localSize)),
        ready1
      )
      callKernel(
        program, "reduce",
        KernelArg(resBuffer) :: KernelArg(resBuffer) :: KernelArg(null, Sizeof.cl_double * localSize) :: KernelArg(numWorkGroups) :: Nil,
        Array(ready1),
        Dimensions(1, Array(0), Array(localSize), Array(localSize)),
        ready2
      )
      val future = new CompletableFuture[Double]
      val result = clEnqueueMapBuffer(queue, resBuffer, false, CL_MAP_READ, 0, Sizeof.cl_double, 1, Array(ready2), finished, null)
      clSetEventCallback(finished, CL_COMPLETE, new EventCallbackFunction(){
        clRetainMemObject(resBuffer)
        override def function(event: cl_event, command_exec_callback_type: Int, user_data: AnyRef): Unit = {
          future.complete(result.order(ByteOrder.nativeOrder).asDoubleBuffer.get(0))
          clEnqueueUnmapMemObject(queue, resBuffer, result, 0, null, null)
          clReleaseMemObject(resBuffer)
        }
      }, null)
      future
    } finally {
      clReleaseEvent(finished)
      clReleaseEvent(ready1)
      clReleaseEvent(ready2)
      clReleaseMemObject(resBuffer)
      val endTime = System.nanoTime
      log.trace("reduce overhead took {}ms", (endTime - startTime)/1e6)
    }
  }

  /**
   * Map one Chunk to a new one. YOU NEED A BARRIER AFTERWARDS!
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
      callKernel(
        program, "map",
        KernelArg(input.handle) :: KernelArg(handle) :: Nil,
        null,
        dimensions,
        null
      )
      new Chunk(input.size, handle)
    } catch {
      case e : Throwable => {
        clReleaseMemObject(handle)
        throw e
      }
    }
  }

  val ALLOC_HOST_PTR_ON_DEVICE = {
    /*
     * This decides if writing to a host mapped ALLOC_HOST_PTR buffer is
     * enough. If false, contents are first written to (pinned)
     * ALLOC_HOST_PTR_ON_DEVICE and then clEnqueueCopyBuffer to an 'ordinary'
     * device buffer. If the first buffer is already on the device this
     * effectively halves the usable accelerator memory. On the other hand just
     * using the first buffer if it does not reside on the device would reduce
     * (memory bound) computations to bus speed.
     * TODO make a better choice here, CL_DEVICE_HOST_UNIFIED_MEMORY is deprecated
     */
    val buffer = Array(0)
    clGetDeviceInfo(device, CL_DEVICE_HOST_UNIFIED_MEMORY, Sizeof.cl_int, Pointer.to(buffer), null)
    val res = buffer(0) != 0
    log.info(s"${if (!res) "not " else ""}using unified memory")
    res
  }
  /**
   * Put the contents of an iterator on the gpu in constant sized chunks (default 256MB size).
   * Chunks have to be closed after use
   */
  def stream(it: Iterator[Double], groupSize: Int = 1024*1024*256) : Seq[Chunk] = {
    val res = new ArrayBuffer[Chunk]
    try {
      while(it.hasNext) {
        var on_device : Option[cl_mem] = None
        var on_host : Option[cl_mem] = None
        val unmapEvent = new cl_event
        try {
          // Allocating direct buffers via ByteBuffer is prone to oom problems
          // it's faster anyway to refcount this
          //val rawBuffer = ByteBuffer.allocateDirect(groupSize)
          on_host = Some(clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, groupSize, null, null))
          val rawBuffer = clEnqueueMapBuffer(queue, on_host.get, true, CL_MAP_WRITE, 0, groupSize, 0, null, null, null)
          val buffer = rawBuffer.order(ByteOrder.nativeOrder).asDoubleBuffer
          var copied = 0
          while(copied < groupSize/Sizeof.cl_double && it.hasNext) {
            buffer.put(copied, it.next)
            copied += 1
          }
          clEnqueueUnmapMemObject(queue, on_host.get, rawBuffer, 0, null, unmapEvent)
          
          if(ALLOC_HOST_PTR_ON_DEVICE) {
            on_device = on_host
            on_host = None
          } else {
            on_device = Some(clCreateBuffer(context, CL_MEM_READ_ONLY, copied*8, null, null))
            clEnqueueCopyBuffer(queue, on_host.get, on_device.get, 0, 0, copied*8, 1, Array(unmapEvent), null)
          }
          res += Chunk(copied, on_device.get)
        } finally {
          on_device.foreach(clReleaseMemObject)
          on_host.foreach(clReleaseMemObject)
          clReleaseEvent(unmapEvent)
        }
      }
      clEnqueueBarrierWithWaitList(queue, 0, null, null)
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
