package de.qaware.chronix.cl

import org.jocl._
import java.nio.ByteOrder
import java.nio.ByteBuffer

trait CLType[T] extends Serializable {
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
  
  val zeroName: String
}
