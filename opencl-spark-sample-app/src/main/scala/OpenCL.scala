import org.jocl.CL._
import org.jocl._
import org.slf4j.LoggerFactory
import scala.collection.mutable.ArrayBuffer
import java.util.concurrent.ConcurrentHashMap
import java.lang.ref.{ReferenceQueue, WeakReference}
import java.util.concurrent.{Future|FutureTask}

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
        val queue = clCreateCommandQueue(context, device, 0, null) //deprecated for OpenCL 2.0, needed for 1.2!
        new OpenCLSession(context, queue)
      })
    })
  }

  override protected def initialValue = {
    val threadId = java.lang.Thread.currentThread.getId
    devices((threadId % devices.size.toLong).toInt)()
  }
}

object OpenCLSession {
}
  
class Chunk (
  val size: Int,
  val handle: cl_mem,
  val arrived: cl_event
)
  extends java.io.Closeable
{
  override def close : Unit = {
    clReleaseMemObject(handle)
    clReleaseEvent(arrived)
  }
}

class OpenCLSession (val context: cl_context, val queue: cl_command_queue)
{
  lazy val log = LoggerFactory.getLogger(getClass)
  log.info("created OpenCLSession")

  def buildProgram(source: string) : Future[cl_program] = {

  }

  /**
   * Put the contents of an iterator on the gpu in constant sized chunks (default 64MB size).
   */
  def stream(it: Iterator[Double], groupSize: Int = 1024*1024*64) : Seq[Chunk] = {
    val res = new ArrayBuffer[Chunk]
    try {
      while(it.hasNext) {
        var chunk: Option[Chunk] = None
        try {
          val handle: cl_mem = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, groupSize, null, null)
          chunk = Some(new Chunk(groupSize, handle, new cl_event))
          val buffer = clEnqueueMapBuffer(queue, handle, true, CL_MAP_WRITE, 0, groupSize, 0, null, null, null)
          var copied = 0
          while((copied + 1)*8 <= groupSize && it.hasNext) {
            buffer.putDouble(copied, it.next)
            copied += 1
          }
          clEnqueueUnmapMemObject(queue, handle, buffer, 0, null, chunk.get.arrived)
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
