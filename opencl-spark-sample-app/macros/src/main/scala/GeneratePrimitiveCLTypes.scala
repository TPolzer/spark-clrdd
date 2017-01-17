package de.qaware.chronix.cl.macros

import scala.reflect.macros.whitebox.Context
import scala.language.experimental.macros
import scala.annotation.StaticAnnotation
import scala.annotation.compileTimeOnly

/** Generate instances of CLType[T] at compile time
 *
 * This macro generates instances for all primitive types and their
 * corresponding tuple(scala)/vector(opencl) types automatically. It is keyed
 * to only match on an empty "object ImplicitCLTypes", which will be replaced
 * by implicit instances of CLType[T]. Needs Macro Paradise:
 * http://docs.scala-lang.org/overviews/macros/paradise.html
 */
@compileTimeOnly("enable macro paradise to expand macro annotations")
class GeneratePrimitiveCLTypes extends StaticAnnotation {
  def macroTransform(annottees: Any*): Any = macro CLTypeGenerator.generate
}

private[macros] class CLTypeGenerator(val c: Context) {
  import c.universe._
  val typeMap = Map(
    tq"Byte" -> "char",
    tq"Char" -> "short",
    tq"Int" -> "int",
    tq"Long" -> "long",
    tq"Float" -> "float",
    tq"Double" -> "double"
  )
  val extensions = Map(
    "char" -> "cl_khr_byte_addressable_store",
    "double" -> "cl_khr_fp64"
  )
  val arities = List(1,2,4,8,16)
  def generate(annottees: c.Expr[Any]*) : c.Expr[Any] = {
    annottees map (_.tree) toList match {
      case q"object ImplicitCLTypes" :: Nil => {
        val declarations =
          (for((scala_t1, opencl_t1) <- typeMap; n <- arities) yield {
            val nid = if (n > 1) n.toString else ""
            val scala_t = tq"(..${List.fill(n)(scala_t1)})"
            val opencl_t = opencl_t1 + nid
            val stringName = s"CL$scala_t1" + nid
            val instanceName = TypeName(stringName)
            val bbName = if(opencl_t1 == "char") "" else scala_t1
            def fromTuple(v: String, offset: Int) = {
              if(n == 1) q"${TermName(v)}"
              else q"${TermName(v)}.${TermName("_" + (offset+1).toString)}"
            }
            List(q"""trait $instanceName extends CLType[$scala_t] {
              override val clName = $opencl_t
              override type doubleCLType = ${TypeName("CLDouble" + nid)}
              override val doubleCLInstance = ${TermName("CLDouble" + nid)}
              override val selfInstance = ${TermName(stringName)}
              override val elemClassTag = implicitly[ClassTag[$scala_t]]
              override val zeroName = ${"((" + opencl_t + ")0)"}
              ${extensions.get(opencl_t1).map(extension =>
                q"override def header = ${"#pragma OPENCL EXTENSION " + extension + " : enable\n"}")
                .getOrElse(q"")
              }
              override val sizeOf = Sizeof.${TermName("cl_" + opencl_t)}
              private final val singleSize = Sizeof.${TermName("cl_" + opencl_t1)}
              override def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer) : $scala_t = {
                (..${(0 to n-1).map(offset =>
                  q"rawBuffer.order(ByteOrder.nativeOrder).${TermName("get" + bbName)}(($n * idx + $offset) * singleSize)"
                )})
              }
              override def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: $scala_t) : Unit = {
                ..${(0 to n-1).map(offset =>
                  q"rawBuffer.order(ByteOrder.nativeOrder).${TermName("put" + bbName)}(($n * idx + $offset) * singleSize, ${fromTuple("v", offset)})"
                )}
              }
            }""",
            q"implicit object ${TermName(stringName)} extends $instanceName") ++
            (if(n > 1)
              List(q"""
                implicit object ${TermName(stringName + "Numeric")} extends Numeric[$scala_t] {
                  private type T = $scala_t
                  private val nt = implicitly[Numeric[$scala_t1]]
                  override def fromInt(x: Int): T = (..${List.fill(n)(q"nt.fromInt(x)")})
                  override def negate(x: T): T =
                    (..${(0 to n-1).map(i => q"nt.negate(${fromTuple("x", i)})")})
                  override def minus(x: T,y: T): T = plus(x, negate(y))
                  override def plus(x: T,y: T): T =
                    (..${(0 to n-1).map(i => q"nt.plus(${fromTuple("x", i)}, ${fromTuple("y", i)})")})
                  override def compare(x: T,y: T): Int = implicitly[Ordering[T]].compare(x, y)
                  override def times(x: T,y: T): T =
                    (..${(0 to n-1).map(i => q"nt.times(${fromTuple("x", i)}, ${fromTuple("y", i)})")})
                  override def toDouble(x: T): Double = ???
                  override def toFloat(x: T): Float = ???
                  override def toInt(x: T): Int = ???
                  override def toLong(x: T): Long = ???
                }
              """)
            else
              Nil
            )
          }).flatten
        val res = c.Expr[Any](q"..$declarations")
        res
      }
      case _ => c.abort(c.enclosingPosition, "invalid application of @GeneratePrimitiveCLTypes")
    }
  }
}
