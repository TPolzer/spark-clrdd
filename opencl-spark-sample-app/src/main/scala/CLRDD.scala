import org.apache.spark.rdd._
import org.apache.spark._

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
    partition.next.copyOrGet(ctx).iterator
  }

  override protected def getPartitions : Array[Partition] = {
    wrapped.partitions
  }

  override protected def getDependencies : Seq[Dependency[_]] = {
    Array(new OneToOneDependency(wrapped), new OneToOneDependency(parentRDD))
  }
}

class CLDoublePartition (val parentPartition: Partition, val parentRDD: RDD[Double])
{
  @transient private var storage : Array[Double] = null
  def copyOrGet (ctx: TaskContext) = {
    if (storage == null) {
      storage = parentRDD.iterator(parentPartition, ctx).toArray
    }
    storage
  }
}
