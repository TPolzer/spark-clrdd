package de.qaware.chronix

import scala.collection.Map

abstract class CLProgramSource extends Product /*with HashcodeCaching*/ {
  val fp64 = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
  def accessors = Map.empty[CLType[_], String]
  def accessor[T]()(implicit clT: CLType[T]) = {
    (clT -> s"__chronix_${clT.clName}")
  }
  def header = {
    Iterator(fp64) ++ accessors.iterator.flatMap({case (clT : CLType[_], id: String) => {
      Iterator(
        s"inline ${clT.clName} ${id}_get(__global char *buffer, long i) {\n",
        "  ", clT.getter, "\n",
        "}\n",
        s"inline void ${id}_set(${clT.clName} v, __global char *buffer, long i) {\n",
        "  ", clT.setter, "\n",
        "}\n"
      )
    }})
  }
  def set[T: CLType](a : String, i : String, v : String) =
    accessors(implicitly[CLType[T]]) ++ s"_set($v, $a, $i)"
  def get[T: CLType](a : String, i : String) =
    accessors(implicitly[CLType[T]]) ++ s"_get($a, $i)"
  def generateSource : Array[String]
}

object CLProgramSource {
  def freshSupply = Iterator.from(0)
    .map(i => s"__chronix_generated_$i")
}

case class MapReduceKernel[A, B](
  f: MapKernel[A,B],
  reduceBody: String,
  identity: String,
  cpu: Boolean,
  implicit val clA: CLType[A],
  implicit val clB: CLType[B]
) extends CLProgramSource {
  override def accessors = super.accessors + accessor[A] + accessor[B] ++ f.accessors
  def genReduceFunction(fresh_ids: Iterator[String]) : (Iterator[String], String) = {
    val r = fresh_ids.next()
    (Iterator(
      s"inline ${clB.clName} $r(${clB.clName} x, ${clB.clName} y) {\n",
        "  ", reduceBody, "\n",
      "}\n"
    ), r)
  }
  def main(r: String, f:String) = Iterator(
    "__kernel\n",
    s"__attribute__((vec_type_hint(${clA.clName})))\n",
    "void reduce(__global char *restrict input, __global char *restrict output, __local char *restrict scratch, long size) {\n") ++ (if(cpu) Iterator(
      s"${clB.clName} cur = $identity\n",
      "for(long i=0; i<size; ++i) {\n",
      s"  cur = $r(cur, $f(", get[A]("input", "i"), "))\n",
      "}\n",
      set[B]("output", "cur", "0"), ";\n",
      "}"
    ) else Iterator(
      "int tid = get_local_id(0);\n",
      "long i = get_group_id(0) * get_local_size(0) + get_local_id(0);\n",
      "long gridSize = get_local_size(0) * get_num_groups(0);\n",
      s"${clB.clName} cur = $identity;\n",
      "for (; i<size; i = i + gridSize){\n",
      s"  cur = $r(cur, $f(", get[A]("input", "i"), "));\n",
      "}\n",
      set[B]("scratch", "tid", "cur"), ";\n",
      "barrier(CLK_LOCAL_MEM_FENCE);\n",
      "for (int s = get_local_size(0) / 2; s>0; s = s >> 1){\n",
      "  if (tid<s){\n",
      set[B]("scratch", "tid", s"$r(${get[B]("scratch", "tid")}, ${get[B]("scratch", "tid+s")})"), ";\n",
      "  }\n",
      "  barrier(CLK_LOCAL_MEM_FENCE);\n",
      "}\n",
      "if (tid==0){\n",
      "  ", set[B]("output", "get_group_id(0)", get[B]("scratch", "0")), ";\n",
      "}\n",
      "}\n"
      ))
  override def generateSource = {
    val supply = CLProgramSource.freshSupply
    val (fsrc, fsymb) = f.genMapFunction(supply)
    val (rsrc, rsymb) = genReduceFunction(supply)
    header ++ fsrc ++ rsrc ++ main(rsymb, fsymb)
  }.toArray
}

abstract class MapKernel[A, B]()(implicit clA: CLType[A], clB: CLType[B]) extends CLProgramSource {
  def genMapFunction(fresh_ids: Iterator[String]) : (Iterator[String], String) // Source code, function symbol
  override def generateSource = {
    val supply = CLProgramSource.freshSupply
    val (code, f) = genMapFunction(supply)
    header ++ code ++ main(f)
  }.toArray
  override def accessors = super.accessors + accessor[A] + accessor[B]
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
  override def accessors = super.accessors ++ kernel.accessors
  override def genMapFunction(fresh_ids: Iterator[String]) = kernel.genMapFunction(fresh_ids)
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
  override def accessors = super.accessors ++ f.accessors ++ g.accessors
  override def genMapFunction(fresh_ids: Iterator[String]) = {
    val (fsrc, fsymb) = f.genMapFunction(fresh_ids)
    val (gsrc, gsymb) = g.genMapFunction(fresh_ids)
    val h = fresh_ids.next()
    (fsrc ++ gsrc ++ Iterator(
      s"inline ${clC.clName} $h(${clA.clName} x) {\n",
      s"  return $gsymb($fsymb(x));\n",
      "}\n"
    ), h)
  }
}

case class MapFunction[A: CLType, B: CLType](
  body: String,
  clA: CLType[A], clB: CLType[B]
) extends MapKernel[A,B]
{
  override def genMapFunction(fresh_ids: Iterator[String]) = {
    val f = fresh_ids.next()
    (Iterator(
      s"inline ${clB.clName} $f(${clA.clName} x) {\n",
      "  ", body, "\n",
      "}\n"
    ), f)
  }
}
  

trait HashcodeCaching { self: Product =>
  override lazy val hashCode: Int = {
    scala.runtime.ScalaRunTime._hashCode(this)
  }
}
