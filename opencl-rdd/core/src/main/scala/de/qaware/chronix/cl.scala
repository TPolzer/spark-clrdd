package de.qaware.chronix

import cl.macros._

import scala.reflect.ClassTag

import org.jocl._
import java.nio.ByteOrder
import java.nio.ByteBuffer


package object cl {
  /*
   * This is a placeholder object for the macro to consume.
   * It will put implicit CLType instances here, see its source.
   */
  @GeneratePrimitiveCLTypes
  object ImplicitCLTypes
}
