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
    val res = partition.next.iterator(ctx)
    partition.hasNext //free spark datastructures
    res
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

trait CLPartition[T] { self =>
  @transient protected var session : OpenCLSession = null
  @transient protected var storage : Array[OpenCL.Chunk[T]] = null
  protected var doCache = false
  protected def src : (OpenCLSession, Iterator[OpenCL.Chunk[T]])
  def get : (OpenCLSession, Iterator[OpenCL.Chunk[T]]) = {
    if(doCache) {
      if(storage == null) {
        val res = src
        session = res._1
        storage = res._2.toArray
      }
      (session, storage.iterator)
    } else {
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
      session = null
    }
  }
  def sum(implicit evidence: T =:= Double) : Double = {
    val (session, chunks) = get
    chunks.map(chunk => {
      try {
        session.reduceChunk[Double](chunk.asInstanceOf[OpenCL.Chunk[Double]], "", "0", "return x+y;")
      } finally {
        if(!doCache) chunk.close
      }
    }).foldLeft(0.0)({case (x,f) => x + f.get})
  }
  def map[B](functionBody: String)(implicit clA: CLType[T], clB: CLType[B]) : CLPartition[B] = {
    new CLPartition[B](){
      override def src = {
        val (session, parentChunks) = self.get
        val mappedChunks = parentChunks.map(c => {
            if(self.doCache) {
              session.mapChunk[T,B](c, functionBody, false)
            } else {
              session.mapChunk[T,B](c, functionBody, true)
            }
        })
        (session, mappedChunks)
      }
    }
  }
}


class CLDoublePartition (val parentPartition: Partition, val parentRDD: RDD[Double])
  extends CLPartition[Double]
{
  lazy val log = LoggerFactory.getLogger(getClass)
  override def src : (OpenCLSession, Iterator[OpenCL.Chunk[Double]]) = {
    val ctx = TaskContext.get
    val session = OpenCL.get
    (session, session.stream(parentRDD.iterator(parentPartition, ctx)))
  }
  def iterator (ctx: TaskContext) : Iterator[Double] = {
    val (session, chunks) = get
    chunks.flatMap(chunk => session.ChunkIterator(chunk))
  }
}
