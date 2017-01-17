package de.qaware.chronix.cl

import org.jocl._
import org.jocl.CL._

import java.util.concurrent.atomic.AtomicInteger

import org.slf4j.LoggerFactory

object OpenCL
{
  private lazy val log = LoggerFactory.getLogger(getClass)
  setExceptionsEnabled(true)
  var CPU = false
  lazy val devices = for(cpu <- Array(false, true)) yield { // holds one session for each device
    val deviceType = if(cpu) CL_DEVICE_TYPE_CPU else CL_DEVICE_TYPE_GPU
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
          (try {
            val maxComputeUnits = Array(0)
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 4, Pointer.to(maxComputeUnits), null)
            var partitionSize = maxComputeUnits(0)
            for(i <- ((maxComputeUnits(0)+31)/32 to maxComputeUnits(0)-1).reverse) { // create at most 32 subdevices
              if(maxComputeUnits(0) % i == 0)
                partitionSize = i
            }
            val splitProperties = new cl_device_partition_property
            splitProperties.addProperty(CL_DEVICE_PARTITION_EQUALLY, partitionSize)
            clCreateSubDevices(device, splitProperties, 0, null, numDevices)
            val devices = new Array[cl_device_id](numDevices(0))
            clCreateSubDevices(device, splitProperties, devices.size, devices, null)
            devices
          } catch { case e: CLException =>
            Array(device)
          }).flatMap(device => {
            try{
              val context = clCreateContext(contextProperties, 1, Array(device), null, null, null)
              val queue = clCreateCommandQueue(context, device, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE | CL_QUEUE_PROFILING_ENABLE, null) //deprecated for OpenCL 2.0, needed for 1.2!
              Some(new OpenCLSession(context, queue, device, cpu))
            } catch {
              case e: CLException => {
                log.trace("got {} while looking for devices", e)
                None
              }
            }
          })
        })
      } catch {
        case e: CLException => {
          log.trace("got {} while looking for devices", e)
          Nil
        }
      }
    })
  }

  def safeReleaseEvent(e: cl_event) = if(e != null && e != new cl_event) clReleaseEvent(e)

  val deviceIndex = Array(new AtomicInteger(0), new AtomicInteger(0))

  /*
   * supplies OpenCLSessions round robin over devices
   */
  def get(cpu : Boolean) = {
    val cpuIndex = if(cpu) 1 else 0
    val updateIndex = new java.util.function.IntUnaryOperator() {
      override def applyAsInt(i: Int) = {
        (i + 1) % devices(cpuIndex).size
      }
    }
    val idx = deviceIndex(cpuIndex).getAndUpdate(updateIndex)
    devices(cpuIndex)(idx)
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
    def apply(i: Long) : KernelArg = KernelArg(Pointer.to(Array(i)), Sizeof.cl_long)
    def apply(c: Chunk[_]) : KernelArg =
      if(c.size != 0)
        KernelArg(c.handle)
      else
        KernelArg(new Pointer(), Sizeof.cl_mem)
  }

  case class Dimensions (
    dim: Int,
    global_work_offset: Array[Long],
    global_work_size: Array[Long],
    local_work_size: Array[Long]
  ) {
  }

  case class Chunk[T] (
    val size: Long, //used elements
    val space: Long, //allocated size in bytes
    var handle: cl_mem, // handle == null iff (size == 0 || this is closed)
    var ready: cl_event // ready == null iff this is closed
  )
  extends java.io.Closeable
  {
    if(handle != null)
      clRetainMemObject(handle)
    else
      assert(size == 0)
    clRetainEvent(ready)
    override def close : Unit = {
      LoggerFactory.getLogger(getClass).info("closing chunk {}", this)
      if(handle != null)
        clReleaseMemObject(handle)
      if(ready != null)
        clReleaseEvent(ready)
      handle = null
      ready = null
    }
    override def finalize = {
      if(handle != null) {
        /*
         * If you arrive at this warning often, be advised that it can easily
         * lead to out of memory situations (on the device). Close your chunks.
         */
        LoggerFactory.getLogger(getClass).warn("closing chunk {} on finalization", this)
        close
      }
    }
  }
}
