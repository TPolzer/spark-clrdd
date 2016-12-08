package de.qaware.chronix

import org.jocl._
import org.jocl.CL._

import java.util.concurrent.atomic.AtomicLong

import org.slf4j.LoggerFactory

import OpenCL.safeReleaseEvent

class ProfilingCallback(ready: cl_event, counter: Option[AtomicLong]) extends EventCallbackFunction {
  clRetainEvent(ready)
  override def function(ready: cl_event, command_exec_callback_type: Int, user_data: AnyRef) = {
	try {
	  val profiling = new Array[Long](1)
	  clGetEventProfilingInfo(ready, CL_PROFILING_COMMAND_QUEUED, Sizeof.cl_ulong, Pointer.to(profiling), null)
	  val queued = profiling(0)
	  clGetEventProfilingInfo(ready, CL_PROFILING_COMMAND_START, Sizeof.cl_ulong, Pointer.to(profiling), null)
	  val started = profiling(0)
	  LoggerFactory.getLogger(getClass).trace("{} lingered in queue for {}ms", ready, (started-queued)/1e6)
	  clGetEventProfilingInfo(ready, CL_PROFILING_COMMAND_END, Sizeof.cl_ulong, Pointer.to(profiling), null)
	  val ended = profiling(0)
	  counter.foreach(_.addAndGet(ended - started))
	  LoggerFactory.getLogger(getClass).trace("{} took {}ms", ready, (ended-started)/1e6)
	} finally safeReleaseEvent(ready)
  }
}
