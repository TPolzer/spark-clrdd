package de.qaware.chronix

abstract class CLProgramSource extends Product with HashcodeCaching {
  val fp64 = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
  val accessors = new scala.collection.mutable.HashMap[CLType[_], String]
  def require[T]()(implicit clT: CLType[T]) = {
    accessors += (clT -> s"__chronix_${clT.clName}")
  }
  def header = {
    Iterator(fp64) ++ accessors.iterator.flatMap({case (clT : CLType[_], id: String) => {
      Iterator(
        s"inline ${clT.clName} ${id}_get(__global char *buffer, long i) {\n",
        "  ", clT.getter,
        "}\n",
        s"inline void ${id}_set(${clT.clName} v, __global char *buffer, long i) {\n",
        "  ", clT.setter,
        "}\n"
      )
    }})
  }
  def generateSource : Array[String]
}

object MapKernel {
  private def freshSupply = Iterator.from(0)
    .map(i => s"__chronix_mapGenerated_$i")
}

abstract class MapKernel[A, B]()(implicit clA: CLType[A], clB: CLType[B]) extends CLProgramSource {
  def genMapFunction(fresh_ids: Iterator[String]) : (Iterator[String], String) // Source code, function symbol
  override def generateSource = {
    val supply = MapKernel.freshSupply
    val (code, f) = genMapFunction(supply)
    header ++ code ++ main(f)
  }.toArray
  def main(f: String) : Iterator[String] = Iterator(
    "__kernel __attribute__((vec_type_hint(", implicitly[CLType[B]].clName, ")))\n",
    "void map(__global char *input, __global char *output) {\n",
    "  long i = get_global_id(0);\n",
    s"  ${accessors(clB)}_set($f(${accessors(clA)}_get(input, i)), output, i);\n",
    "}\n"
  )
}

case class InplaceMap[A, B] (
  kernel: MapKernel[A, B]
)(implicit val clA: CLType[A], val clB: CLType[B]) extends MapKernel[A, B]
{
  override def genMapFunction(fresh_ids: Iterator[String]) = kernel.genMapFunction(fresh_ids)
  override val accessors = kernel.accessors
  override def main(f: String) : Iterator[String] = Iterator(
    "__kernel __attribute__((vec_type_hint(", implicitly[CLType[B]].clName, ")))\n",
    "void map(__global char *input) {\n",
    "  long i = get_global_id(0);\n",
    s"  ${accessors(clB)}_set($f(${accessors(clA)}_get(input, i)), input, i);\n",
    "}\n"
  )
}

case class MapComposition[A: CLType, B: CLType, C: CLType] (
  f: MapKernel[A,B],
  g: MapKernel[B,C],
  clA: CLType[A], clB: CLType[B], clC: CLType[C]
) extends MapKernel[A,C]
{
  override def genMapFunction(fresh_ids: Iterator[String]) = ???
}

case class MapFunction[A: CLType, B: CLType](
  body: String,
  clA: CLType[A], clB: CLType[B]
) extends MapKernel[A,B]
{
  require[A]()
  require[B]()
  override def genMapFunction(fresh_ids: Iterator[String]) = {
    val f = fresh_ids.next()
    (Iterator(
      s"inline ${clB.clName} $f(${clA.clName} x) {\n",
        body,
      "}\n"
    ), f)
  }
}
  

trait HashcodeCaching { self: Product =>
  override lazy val hashCode: Int = {
    scala.runtime.ScalaRunTime._hashCode(this)
  }
}
