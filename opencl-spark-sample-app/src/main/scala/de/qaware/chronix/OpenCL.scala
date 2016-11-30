package de.qaware.chronix

import org.jocl._
import org.jocl.CL._

import org.slf4j.LoggerFactory

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
          (try {
            val maxComputeUnits = Array(0)
            clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 4, Pointer.to(maxComputeUnits), null)
            var partitionSize = maxComputeUnits(0)
            for(i <- ((maxComputeUnits(0)+15)/16 to maxComputeUnits(0)-1).reverse) {
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
              Some(new OpenCLSession(context, queue, device))
            } catch {
              case e: CLException => None
            }
          })
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
    def apply(i: Long) : KernelArg = KernelArg(Pointer.to(Array(i)), Sizeof.cl_long)
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
