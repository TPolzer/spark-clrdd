import org.apache.spark.rdd._
import org.apache.spark._
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._

object CLDoubleRDD
{
  @transient lazy private val sc = SparkContext.getOrCreate
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

  def cacheGPU = {
    wrapped.foreach(_.cache)
    this
  }

  def uncacheGPU = {
    wrapped.foreach(_.uncache)
    this
  }
}

trait CLPartition[T] {
  @transient protected var session : OpenCLSession = null
  @transient protected var storage : Array[OpenCL.Chunk] = null
  protected var doCache = false
  protected def src : Iterator[OpenCL.Chunk]
  def get : Iterator[OpenCL.Chunk] = {
    if(doCache) {
      if(storage == null) {
        session = OpenCL.get
        storage = src.toArray
      }
      storage.iterator
    } else {
      session = OpenCL.get
      src
    }
  }
  def cache = {
    doCache = true
  }
  def uncache = {
    if(storage != null) {
      doCache = false
      storage.foreach(_.close)
      storage = null
    }
  }
  def sum(implicit evidence: T =:= Double) : Double = {
    get.map(c => {
      try {
        session.reduceChunk(c, "0", "return x+y;")
      } finally {
        if(!doCache) c.close
      }
    }).foldLeft(0.0)({case (x,f) => x + f.get})
  }
}


class CLDoublePartition (val parentPartition: Partition, val parentRDD: RDD[Double])
  extends CLPartition[Double]
{
  lazy val log = LoggerFactory.getLogger(getClass)
  override def src : Iterator[OpenCL.Chunk] = {
    val ctx = TaskContext.get
    session.stream(parentRDD.iterator(parentPartition, ctx))
  }
  def iterator (ctx: TaskContext) : Iterator[Double] = {
    get.flatMap(chunk => session.ChunkIterator(chunk))
  }
}
