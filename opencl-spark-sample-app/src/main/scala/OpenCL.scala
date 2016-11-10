import org.jocl.CL._
import org.jocl._
import org.slf4j.LoggerFactory
import scala.collection.mutable.ArrayBuffer
import java.util.WeakHashMap
import java.lang.ref.{ReferenceQueue, WeakReference}
import java.util.concurrent.{Future, CompletableFuture}
import java.nio.ByteOrder

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
        val queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, null) //deprecated for OpenCL 2.0, needed for 1.2!
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

object Test {
  def main(args: Array[String]) = {
    val chunk = OpenCL.get.stream((1 to 10).view.map(_.toDouble).iterator)(0)
    val mapped = OpenCL.get.mapChunk(chunk, "return x*x;")
    val b = clEnqueueMapBuffer(OpenCL.get.queue, mapped.handle, true, CL_MAP_READ, 0, 10*Sizeof.cl_double, 1, Array(mapped.ready), null, null).order(ByteOrder.nativeOrder).asDoubleBuffer
    println((0 to 9).map(b.get(_)))
    clWaitForEvents(1, Array(mapped.ready))
    println((0 to 9).map(b.get(_)))
  }
}

class OpenCLSession (val context: cl_context, val queue: cl_command_queue)
{
  import OpenCL.{Program, Chunk, KernelArg, Dimensions}
  val log = LoggerFactory.getLogger(getClass)
  log.info("created OpenCLSession")


  /*
   * The easy way out: Should keep most programs for long enough if not under
   * severe memory pressure.
   */
  private val programCache = java.util.Collections.synchronizedMap(new WeakHashMap[Seq[String], Program])
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

  def callKernel(program: Program, kernelName: String, args: Seq[KernelArg], dependencies: Seq[cl_event], dimensions: Dimensions) : cl_event = {
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

  def reduceChunk(input: Chunk, identityElement: String, reduceBody: String): Double = {
    val program = getProgram(Array(
      "inline double f(double x, double y) {",
      reduceBody,
      "}",
      "__kernel __attribute__((vec_type_hint(double)))",
      "void reduce(__global double *input, __global double *output, __local double *scratch, int size) {",
        "int tid = get_local_id(0);",
        "int i = get_group_id(0) * get_local_size(0) + get_local_id(0);",
        "int gridSize = get_local_size(0) * get_num_groups(0);",
        s"double cur = $identityElement;",
        "for (; i<size; i = i + gridSize){",
          "cur = f(cur, input[i]);",
        "}",
        "scratch[tid]  = cur;",
        "barrier(CLK_LOCAL_MEM_FENCE);",
        "for (int s = get_local_size(0) / 2; s>0; s = s >> 1){",
          "if (tid<s){",
            "scratch[tid]  = f(scratch[tid], scratch[(tid + s)]);",
          "}",
          "barrier(CLK_LOCAL_MEM_FENCE);",
        "}",
        "if (tid==0){",
          "output[get_group_id(0)]  = scratch[0];",
        "}",
        "return;",
      "}").map(_++"\n"))
    val numWorkGroups = 8*1024
    val localSize = 64 // TODO make this tunable / evaluate...?
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
    val result = Array(0.0)
    clEnqueueReadBuffer(queue, resBuffer, true, 0, Sizeof.cl_double, Pointer.to(result), 1, Array(ready2), null)
    clReleaseEvent(ready1) // TODO proper error handling!
    clReleaseEvent(ready2)
    clReleaseMemObject(resBuffer)
    result(0)
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
      "}").map(_++"\n"))
    val dimensions = Dimensions(1, Array(0), Array(input.size), null)
    val handle: cl_mem = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, input.size*Sizeof.cl_double, null, null)
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
        var chunk: Option[Chunk] = None
        try {
          val handle: cl_mem = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, groupSize, null, null)
          chunk = Some(new Chunk(0, handle, new cl_event))
          val rawBuffer = clEnqueueMapBuffer(queue, handle, true, CL_MAP_WRITE, 0, groupSize, 0, null, null, null)
          val buffer = rawBuffer.order(ByteOrder.nativeOrder).asDoubleBuffer
          var copied = 0
          while(copied < groupSize/Sizeof.cl_double && it.hasNext) {
            buffer.put(copied, it.next)
            copied += 1
          }
          chunk = chunk.map(_.copy(size = copied))
          clEnqueueUnmapMemObject(queue, handle, rawBuffer, 0, null, chunk.get.ready)
          res += chunk.get
        } catch {
          case e:Throwable => {
            chunk.foreach(_.close)
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
