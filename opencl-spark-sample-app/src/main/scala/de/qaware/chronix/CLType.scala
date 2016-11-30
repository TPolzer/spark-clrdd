package de.qaware.chronix

import org.jocl._
import java.nio.ByteOrder
import java.nio.ByteBuffer

trait CLType[T] extends Numeric[T] {
  val clName: String
  val sizeOf: Int
  val header: String = ""
  val zeroName: String
  def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer): T
  def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: T): Unit
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
