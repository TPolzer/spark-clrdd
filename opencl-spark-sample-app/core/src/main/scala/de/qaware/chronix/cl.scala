package de.qaware.chronix

import cl.macros._

import scala.reflect.ClassTag

import org.jocl._
import java.nio.ByteOrder
import java.nio.ByteBuffer


package object cl {
  @GeneratePrimitiveCLTypes
  object ImplicitCLTypes
}
