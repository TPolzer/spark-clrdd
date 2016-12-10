package de.qaware.chronix

import org.jocl._
import java.nio.ByteOrder
import java.nio.ByteBuffer

trait CLType[T] {
  val clName: String
  val sizeOf: Int
  def header: String = "" // has to be idempotent, define struct here
  def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer): T
  def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: T): Unit
 

  //CLTypes have to be singleton types, this is natural and makes caching kernels easier
  override def equals(o: Any) = {
    o.getClass.equals(this.getClass)
  }
  override def hashCode() = {
    this.getClass.hashCode
  }
}

trait CLNumeric[T] extends CLType[T] with Numeric[T] {
  val zeroName = s"(($clName)0)"
}

/*
 * The following is a huge amount of boilerplate that could theoretically be
 * generated at compile time. 
 */
object CLType {
  trait CLFloat extends CLNumeric[Float] {
    override val clName = "float"
    override val zeroName = "0.0f"
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
    override def header = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
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
  trait CLByte extends CLNumeric[Byte] {
    override val clName = "char"
    override val sizeOf = Sizeof.cl_char
    override def header = "#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable\n"
    override def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer) = {
      rawBuffer.order(ByteOrder.nativeOrder).get(idx * sizeOf)
    }
    override def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: Byte) = {
      rawBuffer.order(ByteOrder.nativeOrder).put(idx * sizeOf, v)
    }
  }
  implicit object CLByte extends CLByte with Numeric.ByteIsIntegral with Ordering.ByteOrdering
  trait CLChar extends CLNumeric[Char] {
    override val clName = "short"
    override val zeroName = "((short)0)"
    override val sizeOf = Sizeof.cl_short
    override def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer) = {
      rawBuffer.order(ByteOrder.nativeOrder).getChar(idx * sizeOf)
    }
    override def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: Char) = {
      rawBuffer.order(ByteOrder.nativeOrder).putChar(idx * sizeOf, v)
    }
  }
  implicit object CLChar extends CLChar with Numeric.CharIsIntegral with Ordering.CharOrdering
  class T2Numeric[T: Numeric] extends Numeric[(T,T)] {
    import Numeric.Implicits._
    def nt = implicitly[Numeric[T]]
    override def fromInt(x: Int): (T, T) = (nt.fromInt(x), nt.fromInt(x))
    override def negate(x: (T, T)): (T, T) = (-x._1, -x._2)
    override def minus(x: (T, T),y: (T, T)): (T, T) = x + (-y)
    override def plus(x: (T, T),y: (T, T)): (T, T) = (x._1 + y._1, x._2 + y._2)
    override def times(x: (T, T),y: (T, T)): (T, T) = ???
    override def toDouble(x: (T, T)): Double = ???
    override def toFloat(x: (T, T)): Float = ???
    override def toInt(x: (T, T)): Int = ???
    override def toLong(x: (T, T)): Long = ???
    override def compare(x: (T, T),y: (T, T)): Int = implicitly[Ordering[(T,T)]].compare(x, y)
  }
  class CLVector2[T: Numeric](implicit clT: CLType[T]) extends T2Numeric[T] with CLNumeric[(T,T)] with Serializable {
    override def header = clT.header
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
  implicit val CLByte2  = new CLVector2[Byte]
  implicit val CLChar2  = new CLVector2[Char]
  implicit val CLInt2 = new CLVector2[Int]
  implicit val CLLong2  = new CLVector2[Long]
}
