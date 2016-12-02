package de.qaware.chronix

import org.jocl._
import java.nio.ByteOrder
import java.nio.ByteBuffer

trait CLType[T] {
  val clName: String
  val sizeOf: Int
  val header: String = ""
  lazy val getter: String = s"return as_$clName(vload$sizeOf(i, buffer));"
  lazy val setter: String = s"vstore$sizeOf(as_char$sizeOf(v), i, buffer);"
  def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer): T
  def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: T): Unit
}

trait CLNumeric[T] extends CLType[T] with Numeric[T] {
  val zeroName: String
}

object CLType {
  trait CLFloat extends CLNumeric[Float] {
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
  trait CLDouble extends CLNumeric[Double] {
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
  trait CLLong extends CLNumeric[Long] {
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
  trait CLInt extends CLNumeric[Int] {
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
  class CLVector2[T](implicit clT: CLType[T]) extends CLType[(T,T)] with Serializable {
    override val clName = clT.clName + "2"
    override val sizeOf = clT.sizeOf * 2
    override def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer) = {
      (clT.fromByteBuffer(idx * 2, rawBuffer), clT.fromByteBuffer(idx * 2 + 1, rawBuffer))
    }
    override def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: (T,T)) = {
      clT.toByteBuffer(idx * 2, rawBuffer, v._1)
      clT.toByteBuffer(idx * 2, rawBuffer, v._2)
    }
  }
  implicit val CLFloat2 = new CLVector2[Float]
  implicit val CLDouble2 = new CLVector2[Double]
  implicit val CLInt2 = new CLVector2[Int]
  implicit val CLLong2  = new CLVector2[Long]
}
