package de.qaware.chronix.cl

abstract class CLProgramSource extends Product with Serializable {
  val fp64 = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
  def accessors = Set.empty[CLType[_]]
  def header = {
    Iterator(fp64) ++
      accessors.iterator.map(_.header)
  }
  def generateSource(supply: Iterator[String]) : Array[String]
}

case class WindowReduction[A, B](
  reduceBody: String,
  implicit val clA: CLType[A],
  implicit val clB: CLType[B]
) extends CLProgramSource {
  override def accessors = super.accessors + clA + clB
  def A = clA.clName
  def B = clB.clName
  def genReduceFunction(supply: Iterator[String]) = {
    val r = supply.next()
    val g = supply.next()
    (Iterator(
      s"inline $A $g(int i, long idx, const __global $A *primary, long primarySize, const __global $A *secondary, int width, int stride, int offset) {\n",
      "  long iabs = idx*stride + offset + i;\n",
      "  if(iabs < primarySize)\n",
      "    return primary[iabs];\n",
      "  else\n",
      "    return secondary[iabs - primarySize];\n",
      "}\n",
      s"inline $B $r(long idx, const __global $A *primary, long primarySize, const __global $A *secondary, int width, int stride, int offset) {\n",
      s"#define GET(i) $g(i, idx, primary, primarySize, secondary, width, stride, offset)\n",
      "  ", reduceBody, "\n",
      "#undef GET\n",
      "}\n"
    ), r)
  }
  def main(r: String) = Iterator(
    "__kernel\n",
    s"__attribute__((vec_type_hint($A)))\n",
    s"void reduce(const __global $A *restrict input, const __global $A *restrict fringe, __global $B *restrict output, long outputSize, long inputSize, int width, int stride, int offset) {\n",
    "  long i = get_global_id(0);\n",
    "  if (i < outputSize)\n",
    s"   output[i] = $r(i, input, inputSize, fringe, width, stride, offset);\n",
    "}\n"
  )
  override def generateSource(supply: Iterator[String]) = {
    val (rsrc, rsymb) = genReduceFunction(supply)
    (header ++ rsrc ++ main(rsymb)).toArray
  }
}

case class MapReduceKernel[A, B](
  f: MapKernel[A,B],
  reduceBody: String,
  identity: String,
  cpu: Boolean,
  implicit val clA: CLType[A],
  implicit val clB: CLType[B]
) extends CLProgramSource {
  override def accessors = super.accessors + clA + clB ++ f.accessors
  def A = clA.clName
  def B = clB.clName
  def genReduceFunction(fresh_ids: Iterator[String]) : (Iterator[String], String) = {
    val r = fresh_ids.next()
    (Iterator(
      s"inline $B $r($B x, $B y) {\n",
        "  ", reduceBody, "\n",
      "}\n"
    ), r)
  }
  def main(r: String, f:String) = Iterator(
    "__kernel\n",
    s"__attribute__((vec_type_hint($A)))\n",
    s"void reduce(__global $A *restrict input, __global $B *restrict output, __local $B *restrict scratch, long size) {\n") ++ (if(cpu) Iterator(
      s"$B cur = $identity;\n",
      "for(long i=0; i<size; ++i) {\n",
      s"  cur = $r(cur, $f(input[i]));\n",
      "}\n",
      "output[0] = cur;\n",
      "}"
    ) else Iterator(
      "int tid = get_local_id(0);\n",
      "long i = get_group_id(0) * get_local_size(0) + get_local_id(0);\n",
      "long gridSize = get_local_size(0) * get_num_groups(0);\n",
      s"$B cur = $identity;\n",
      "for (; i<size; i = i + gridSize){\n",
      s"  cur = $r(cur, $f(input[i]));\n",
      "}\n",
      "scratch[tid] = cur;\n",
      "barrier(CLK_LOCAL_MEM_FENCE);\n",
      "for (int s = get_local_size(0) / 2; s>0; s = s >> 1){\n",
      "  if (tid<s){\n",
      s"scratch[tid] = $r(scratch[tid], scratch[tid+s]);\n",
      "  }\n",
      "  barrier(CLK_LOCAL_MEM_FENCE);\n",
      "}\n",
      "if (tid==0){\n",
      "  output[get_group_id(0)] = scratch[0];\n",
      "}\n",
      "}\n"
      ))
  override def generateSource(supply: Iterator[String]) = {
    val (fsrc, fsymb) = f.genMapFunction(supply)
    val (rsrc, rsymb) = genReduceFunction(supply)
    header ++ fsrc ++ rsrc ++ main(rsymb, fsymb)
  }.toArray
  def precomposeMap[C](m: MapKernel[C,A])(implicit clC: CLType[C]) = {
    MapReduceKernel(
      m.compose(f),
      reduceBody,
      identity,
      cpu,
      clC, clB
    )
  }
  def stage2 =
    this.copy(f = MapKernel.identity[B], clA = clB)
}

object MapKernel {
  def identity[A]()(implicit clA: CLType[A]) = MapFunction[A,A]("return x;", clA, clA)
}

abstract class MapKernel[A, B]()(implicit clA: CLType[A], clB: CLType[B]) extends CLProgramSource {
  def genMapFunction(fresh_ids: Iterator[String]) : (Iterator[String], String) // Source code, function symbol
  override def generateSource(supply: Iterator[String]) = {
    val (code, f) = genMapFunction(supply)
    header ++ code ++ main(f, supply)
  }.toArray
  override def accessors = super.accessors + clA + clB
  def A = clA.clName
  def B = clB.clName
  def main(f: String, fresh_ids: Iterator[String]) : Iterator[String] = Iterator(
    s"__kernel __attribute__((vec_type_hint($B)))\n",
    s"void map(__global $A *input, long inputSize, __global $B *output) {\n",
    "  long i = get_global_id(0);\n",
    "  if(i < inputSize)\n",
    s"   output[i] = $f(input[i]);\n",
    "}\n"
  )
  def compose[C: CLType](g:MapKernel[B,C]) : MapKernel[A,C] =
    if(this == MapKernel.identity[A])
      g.asInstanceOf[MapKernel[A,C]]
    else if(g == MapKernel.identity[B])
      this.asInstanceOf[MapKernel[A,C]]
    else
      MapComposition(this, g, implicitly[CLType[A]], implicitly[CLType[B]], implicitly[CLType[C]])
}

case class InplaceMap[A, B] (
  kernel: MapKernel[A, B]
)(implicit val clA: CLType[A], val clB: CLType[B]) extends MapKernel[A, B]
{
  override def accessors = super.accessors ++ kernel.accessors
  override def genMapFunction(fresh_ids: Iterator[String]) = kernel.genMapFunction(fresh_ids)
  override def main(f: String, fresh_ids: Iterator[String]) : Iterator[String] = {
    if(A == B) {
      Iterator(
        s"__kernel __attribute__((vec_type_hint($B)))\n",
        s"void map(__global $A *input, long inputSize) {\n",
        "  long i = get_global_id(0);\n",
        "  if(i < inputSize)\n",
        s"  input[i] = $f(input[i]);\n",
        "}\n"
      )
    } else {
      val union_t = "union " ++ fresh_ids.next()
      Iterator(
        s"$union_t {\n",
        s"  $A __$A;\n",
        s"  $B __$B;\n",
        "};\n",
        s"__kernel __attribute__((vec_type_hint($B)))\n",
        s"void map(__global $union_t *input, long inputSize) {\n",
        "  long i = get_global_id(0);\n",
        "  if(i < inputSize)\n",
        s"  input[i].__$B = $f(input[i].__$A);\n",
        "}\n"
      )
    }
  }
}

case class MapComposition[A: CLType, B: CLType, C: CLType] (
  f: MapKernel[A,B],
  g: MapKernel[B,C],
  clA: CLType[A], clB: CLType[B], clC: CLType[C]
) extends MapKernel[A,C]
{
  def C = clC.clName
  override def accessors = super.accessors ++ f.accessors ++ g.accessors
  override def genMapFunction(fresh_ids: Iterator[String]) = {
    val (fsrc, fsymb) = f.genMapFunction(fresh_ids)
    val (gsrc, gsymb) = g.genMapFunction(fresh_ids)
    val h = fresh_ids.next()
    (fsrc ++ gsrc ++ Iterator(
      s"inline $C $h($A x) {\n",
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
  override def genMapFunction(fresh_ids: Iterator[String]) : (Iterator[String], String) = {
    if(this == MapKernel.identity[A])
      return (Iterator(), "")
    val f = fresh_ids.next()
    (Iterator(
      s"inline $B $f($A x) {\n",
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
