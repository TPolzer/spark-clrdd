import org.jocl.CL._
import org.jocl._

import org.cache2k.{Cache, Cache2kBuilder}
import org.cache2k.integration.CacheLoader

import org.slf4j.LoggerFactory

import scala.collection.mutable.ArrayBuffer
import scala.concurrent.{Future, Promise}

import scala.math.Numeric
import scala.math.Ordering

import java.util.HashMap
import java.util.PrimitiveIterator
import java.lang.ref.{ReferenceQueue, WeakReference}
import java.util.concurrent.atomic.AtomicLong
import java.nio.ByteOrder
import java.nio.ByteBuffer

object OpenCL
  extends java.lang.ThreadLocal[OpenCLSession] // supplies one OpenCLSession per device, round-robin over threads
{
  setExceptionsEnabled(true)
  var CPU = false
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

  def safeReleaseEvent(e: cl_event) = if(e != null && e != new cl_event) clReleaseEvent(e)

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

  case class Chunk[T] (
    val size: Int, //used elements
    val space: Int, //allocated size in bytes
    var handle: cl_mem,
    var ready: cl_event
  )
  extends java.io.Closeable
  {
    clRetainMemObject(handle)
    clRetainEvent(ready)
    override def close : Unit = {
      if(handle != null)
        clReleaseMemObject(handle)
      if(ready != null)
        clReleaseEvent(ready)
      handle = null
      ready = null
    }
    override def finalize = {
      if(handle != null) {
        LoggerFactory.getLogger(getClass).warn("closing chunk {} on finalization", this)
        close
      }
    }
  }
}

object CLType {
  trait CLFloat extends CLType[Float] {
    override val clName = "float"
    override val zeroName = "0.0"
    override val sizeOf = Sizeof.cl_float
    override def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer) = {
      rawBuffer.order(ByteOrder.nativeOrder).getFloat(idx * sizeOf)
    }
    override def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: Float) = {
      rawBuffer.order(ByteOrder.nativeOrder).putFloat(idx * sizeOf, v)
    }
  }
  implicit object CLFloat extends CLFloat with Numeric.FloatIsConflicted with Ordering.FloatOrdering
  trait CLDouble extends CLType[Double] {
    override val clName = "double"
    override val zeroName = "0.0"
    override val sizeOf = Sizeof.cl_double
    override def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer) = {
      rawBuffer.order(ByteOrder.nativeOrder).getDouble(idx * sizeOf)
    }
    override def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: Double) = {
      rawBuffer.order(ByteOrder.nativeOrder).putDouble(idx * sizeOf, v)
    }
  }
  implicit object CLDouble extends CLDouble with Numeric.DoubleIsConflicted with Ordering.DoubleOrdering
  trait CLLong extends CLType[Long] {
    override val clName = "long"
    override val zeroName = "0L"
    override val sizeOf = Sizeof.cl_long
    override def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer) = {
      rawBuffer.order(ByteOrder.nativeOrder).getLong(idx * sizeOf)
    }
    override def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: Long) = {
      rawBuffer.order(ByteOrder.nativeOrder).putLong(idx * sizeOf, v)
    }
  }
  implicit object CLLong extends CLLong with Numeric.LongIsIntegral with Ordering.LongOrdering
  trait CLInt extends CLType[Int] {
    override val clName = "int"
    override val zeroName = "0"
    override val sizeOf = Sizeof.cl_int
    override def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer) = {
      rawBuffer.order(ByteOrder.nativeOrder).getInt(idx * sizeOf)
    }
    override def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: Int) = {
      rawBuffer.order(ByteOrder.nativeOrder).putInt(idx * sizeOf, v)
    }
  }
  implicit object CLInt extends CLInt with Numeric.IntIsIntegral with Ordering.IntOrdering
}

trait CLType[@specialized(Double, Float, Int, Long) T] extends Numeric[T] {
  val clName: String
  val sizeOf: Int
  val header: String = ""
  val zeroName: String
  def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer): T
  def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: T): Unit
}

class OpenCLSession (val context: cl_context, val queue: cl_command_queue, val device: cl_device_id)
{
  import OpenCL.{Program, Chunk, KernelArg, Dimensions, safeReleaseEvent}
  val log = LoggerFactory.getLogger(getClass)
  log.info("created OpenCLSession")

  case class ChunkIterator[@specialized(Double, Float, Int, Long) T](chunk: Chunk[T])(implicit clT: CLType[T])
    extends Iterator[T] with java.io.Closeable
  {
    private val Chunk(elems, _, handle, inputReady) = chunk
    clRetainMemObject(handle)
    var outputReady = new cl_event
    private var rawBuffer: Option[ByteBuffer] = Some(clEnqueueMapBuffer(queue, handle, false, CL_MAP_READ, 0, clT.sizeOf * elems, 1, Array(inputReady), outputReady, null))
    override def hasNext = {
      val res = (idx != elems)
      if(!res)
        close
      res
    }
    override def next = {
      if(outputReady != new cl_event) {
        clWaitForEvents(1, Array(outputReady))
        safeReleaseEvent(outputReady)
        outputReady = new cl_event
      }
      idx += 1
      clT.fromByteBuffer(idx-1, rawBuffer.get)
    }
    private var idx = 0
    override def close : Unit = {
      rawBuffer.foreach(b => {
        clEnqueueUnmapMemObject(queue, handle, b, 0, null, null)
        clReleaseMemObject(handle)
        safeReleaseEvent(outputReady)
      })
      rawBuffer = None
    }
    override def finalize: Unit = close
  }

  private val programCache = Cache2kBuilder.of(classOf[Seq[String]], classOf[Program])
    .entryCapacity(100)
    .loader(new CacheLoader[Seq[String],Program](){
      override def load(source: Seq[String]) : Program = {
        val program = clCreateProgramWithSource(context, source.size, source.toArray, null, null)
        val res = Program(program) // finalize if buildProgram throws
        clBuildProgram(res.program, 0, null, "-cl-unsafe-math-optimizations", null, null)
        res
      }
    }).build
  /**
   * Build (or get from cache) an OpenCL program.
   */
  def getProgram(source: Seq[String]) : Program = {
    programCache.get(source)
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
      } finally safeReleaseEvent(ready)
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
   * parallelization happens outside Chunks for CPU
   */
  var ngroups = if(OpenCL.CPU) 1 else 8*1024
  var nlocal = if(OpenCL.CPU) 1 else 128
 
  def reduceChunk[T](input: Chunk[T], header: String, identityElement: String, reduceBody: String)(implicit clT: CLType[T]): Future[T] = {
    val elemType = clT.clName
    val startTime = System.nanoTime
    val program = if (OpenCL.CPU)
      getProgram(Array(
      "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
      header,
      "inline ", elemType, " f(", elemType," x, ", elemType, " y) {\n",
      reduceBody,
      """}
      __kernel
      __attribute__((vec_type_hint(""", elemType, """)))
      void reduce(__global """, elemType, """ *input, __global """, elemType, """ *output, __local """, elemType, """ *scratch, int size) {
        if(get_global_id(0) == 0) {
          """, elemType, """ cur = """, identityElement, """;
          for(int i=0; i<size; ++i) {
            cur  = f(cur, input[i]);
          }
          output[0] = cur;
        } else if(get_local_id(0) == 0) {
          output[get_group_id(0)] = """, identityElement, """;
        }
        return;
      }"""))
    else
      getProgram(Array(
      "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n",
      header,
      "inline ", elemType, " f(", elemType," x, ", elemType, " y) {\n",
      reduceBody,
      """}
      __kernel
      __attribute__((vec_type_hint(""", elemType, """)))
      void reduce(__global """, elemType, """ *input, __global """, elemType, """ *output, __local """, elemType, """ *scratch, int size) {
        int tid = get_local_id(0);
        int i = get_group_id(0) * get_local_size(0) + get_local_id(0);
        int gridSize = get_local_size(0) * get_num_groups(0);
        """, elemType, """ cur = """, identityElement, """;
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
    val finished = new cl_event
    val ready1 = new cl_event
    val ready2 = new cl_event
    var reduceBuffer : Option[cl_mem] = None
    var resBuffer : Option[cl_mem] = None
    try {
      reduceBuffer = Some(clCreateBuffer(context, 0, numWorkGroups * clT.sizeOf, null, null))
      resBuffer = Some(clCreateBuffer(context, 0, clT.sizeOf, null, null))
      callKernel(
        program, "reduce",
        KernelArg(input.handle) :: KernelArg(reduceBuffer.get) :: KernelArg(null, clT.sizeOf * localSize) :: KernelArg(input.size) :: Nil,
        Array(input.ready),
        Dimensions(1, Array(0), Array(globalSize), Array(localSize)),
        ready1
      )
      callKernel(
        program, "reduce",
        KernelArg(reduceBuffer.get) :: KernelArg(resBuffer.get) :: KernelArg(null, clT.sizeOf * localSize) :: KernelArg(numWorkGroups) :: Nil,
        Array(ready1),
        Dimensions(1, Array(0), Array(localSize), Array(localSize)),
        ready2
      )
      val promise = Promise[T]
      val future = promise.future
      val result = clEnqueueMapBuffer(queue, resBuffer.get, false, CL_MAP_READ, 0, clT.sizeOf, 1, Array(ready2), finished, null)
      clSetEventCallback(finished, CL_COMPLETE, new EventCallbackFunction(){
        clRetainMemObject(resBuffer.get)
        override def function(event: cl_event, command_exec_callback_type: Int, user_data: AnyRef): Unit = {
          promise.success(clT.fromByteBuffer(0, result))
          clEnqueueUnmapMemObject(queue, resBuffer.get, result, 0, null, null)
          clReleaseMemObject(resBuffer.get)
        }
      }, null)
      future
    } finally {
      safeReleaseEvent(finished)
      safeReleaseEvent(ready1)
      safeReleaseEvent(ready2)
      resBuffer.foreach(clReleaseMemObject)
      reduceBuffer.foreach(clReleaseMemObject)
      val endTime = System.nanoTime
      log.trace("reduce overhead took {}ms", (endTime - startTime)/1e6)
    }
  }

  /**
   * Map one Chunk to a new one.
   * @param functionBody the body of the map function 'inline B f(A x) {#functionBody}'.
   * @param destructive if true, mapping is done in place if possible
   */
  def mapChunk[A,B](input: Chunk[A], functionBody: String, destructive: Boolean = false)(implicit clA: CLType[A], clB: CLType[B]) : Chunk[B] = {
    //could be done in place even if sizes dont line up, but with a lot more complex code
    val inplace = destructive && (clA.sizeOf == clB.sizeOf)
    val dimensions = Dimensions(1, Array(0), Array(input.size), null)
    val ready = new cl_event
    try {
      if(inplace) {
        val program = getProgram(Array(
          """#pragma OPENCL EXTENSION cl_khr_fp64 : enable
        inline """, clB.clName, " f(", clA.clName, " x) {\n",
          functionBody,
        """}
        __kernel __attribute__((vec_type_hint(""", clA.clName, """)))
        void map(__global """, clB.clName, """ *input) {
          int i = get_global_id(0);
          input[i] = f(as_""", clA.clName,"""(input[i]));
        }"""))
        callKernel(
          program, "map",
          KernelArg(input.handle) :: Nil,
          Array(input.ready),
          dimensions,
          ready
        )
        new Chunk[B](input.size, input.space, input.handle, ready)
      } else {
        val program = getProgram(Array(
          """#pragma OPENCL EXTENSION cl_khr_fp64 : enable
        inline """, clB.clName, " f(", clA.clName, " x) {\n",
          functionBody,
        """}
        __kernel __attribute__((vec_type_hint(""", clA.clName, """)))
        void map(__global """, clA.clName, " *input, __global ", clB.clName, """ *output) {
          int i = get_global_id(0);
          output[i] = f(input[i]);
        }"""))
        val handle: cl_mem = clCreateBuffer(context, 0, input.size*clB.sizeOf, null, null)
        try {
          callKernel(
            program, "map",
            KernelArg(input.handle) :: KernelArg(handle) :: Nil,
            Array(input.ready),
            dimensions,
            ready
          )
          new Chunk[B](input.size, input.size*clB.sizeOf, handle, ready)
        } finally {
          clReleaseMemObject(handle)
        }
      }
    } finally {
      safeReleaseEvent(ready)
      if(destructive)
        input.close
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
    val vendorBuffer = new Array[Byte](1024)
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, 1024, Pointer.to(vendorBuffer), null)
    val vendor = new String(vendorBuffer, "UTF-8")
    val res = (buffer(0) != 0) || vendor.toLowerCase.matches(".*nvidia.*")
    log.info(s"${if (!res) "not " else ""}using unified memory")
    res
  }
  /**
   * Put the contents of an iterator on the gpu in constant sized chunks (default 256MB size).
   * Chunks have to be closed after use
   */
  def stream[@specialized(Double, Float, Int, Long) T](it: Iterator[T], groupSize: Int = 1024*1024*64)(implicit clT: CLType[T]) : Iterator[Chunk[T]] = new Iterator[Chunk[T]](){
    override def hasNext = it.hasNext
    override def next = {
      var on_device : Option[cl_mem] = None
      var on_host : Option[cl_mem] = None
      var unmapEvent = new cl_event
      var readyEvent: cl_event = null
      var allocatedSize : Int = 0
      try {
        // Allocating direct buffers via ByteBuffer is prone to oom problems
        // it's faster anyway to refcount this
        //val rawBuffer = ByteBuffer.allocateDirect(groupSize)
        on_host = Some(clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, groupSize, null, null))
        val rawBuffer = clEnqueueMapBuffer(queue, on_host.get, true, CL_MAP_WRITE_INVALIDATE_REGION, 0, groupSize, 0, null, null, null)
        log.trace("mapped buffer {}", on_host.get)
        var copied = 0
        while(copied < groupSize/clT.sizeOf && it.hasNext) {
          clT.toByteBuffer(copied, rawBuffer, it.next)
          copied += 1
        }
        log.trace("unmapping buffer {}", on_host.get)
        clEnqueueUnmapMemObject(queue, on_host.get, rawBuffer, 0, null, unmapEvent)

        if(ALLOC_HOST_PTR_ON_DEVICE) {
          on_device = on_host
          readyEvent = unmapEvent
          unmapEvent = null
          on_host = None
          allocatedSize = groupSize
        } else {
          on_device = Some(clCreateBuffer(context, CL_MEM_READ_ONLY, copied*clT.sizeOf, null, null))
          readyEvent = new cl_event
          clEnqueueCopyBuffer(queue, on_host.get, on_device.get, 0, 0, copied*clT.sizeOf, 1, Array(unmapEvent), readyEvent)
          allocatedSize = copied*clT.sizeOf
        }
        Chunk[T](copied, allocatedSize, on_device.get, readyEvent)
      } finally {
        safeReleaseEvent(readyEvent)
        safeReleaseEvent(unmapEvent)
        on_host.foreach(clReleaseMemObject)
        on_device.foreach(clReleaseMemObject)
      }
    }
  }

  override def finalize = {
    log.info("finalizing OpenCLSession")
    clReleaseCommandQueue(queue)
    clReleaseContext(context)
  }
}
