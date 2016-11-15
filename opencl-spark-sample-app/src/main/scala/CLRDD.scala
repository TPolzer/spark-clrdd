import org.apache.spark.rdd._
import org.apache.spark._
import scala.collection.JavaConverters._

object CLDoubleRDD
{
  lazy private val sc = SparkContext.getOrCreate
  def wrap(wrapped: RDD[Double]) = {
    val partitions = sc.broadcast(wrapped.partitions)
    val partitionsRDD = wrapped.mapPartitionsWithIndex( { case (idx: Int, _) =>
        Iterator(new CLDoublePartition(partitions.value(idx), wrapped)) } ).cache
    new CLDoubleRDD(partitionsRDD, wrapped)
  }
}

class CLDoubleRDD (val wrapped: RDD[CLDoublePartition], val parentRDD: RDD[Double])
  extends RDD[Double](wrapped)
{
  override def compute(split: Partition, ctx: TaskContext) : Iterator[Double] = {
    val partition = wrapped.iterator(split, ctx)
    partition.next.iterator(ctx)
  }

  override protected def getPartitions : Array[Partition] = {
    wrapped.partitions
  }

  override protected def getDependencies : Seq[Dependency[_]] = {
    Array(new OneToOneDependency(wrapped), new OneToOneDependency(parentRDD))
  }
  
  def sum : Double = {
    wrapped.map(_.sum).sum
  }
}

class CLDoublePartition (val parentPartition: Partition, val parentRDD: RDD[Double])
{
  @transient private var session : OpenCLSession = null
  @transient private var storage : Seq[OpenCL.Chunk] = null
  def copyOrGet (ctx: TaskContext) = {
    if (storage == null) {
      session = OpenCL.get
      storage = session.stream(parentRDD.iterator(parentPartition, ctx))
    }
    storage
  }
  def sum : Double = {
    val ctx = TaskContext.get
    copyOrGet(ctx).view.map(c => session.reduceChunk(c, "0", "return x+y;")).foldLeft(0.0)({case (x,f) => x + f.get})
  }
  def iterator (ctx: TaskContext) : Iterator[Double] = {
    copyOrGet(ctx).iterator.flatMap(chunk => session.ChunkIterator(chunk))
  }
  override def finalize : Unit = 
    if(storage != null)
      storage.foreach(_.close)
}
