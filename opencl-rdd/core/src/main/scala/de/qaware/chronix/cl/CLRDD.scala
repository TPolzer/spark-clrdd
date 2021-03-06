package de.qaware.chronix.cl

import org.apache.spark.rdd._
import org.apache.spark._
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._

import scala.reflect.ClassTag

import scala.util.Try
import scala.util.Failure
import scala.util.Success
import scala.concurrent.duration._
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.ExecutionContext.Implicits.global

object CLRDD
{
  @transient private lazy val log = LoggerFactory.getLogger(getClass)

  /** Wrap a Spark for OpenCL computations */
  def wrap[T : ClassTag : CLType](wrapped: RDD[T], expectedPartitionSize: Long, cpu: Boolean) : CLRDD[T] =
    wrap(wrapped, Some(expectedPartitionSize), cpu)
  /** Wrap a Spark for OpenCL computations
   *
   *  The unit of expectedPartitionSize should be bytes. Continuous chunks of
   *  that size will be allocated for each partition (leading to failure if the
   *  size is too large, wasted space if it is nearly of partition size and
   *  ineffiency if it is too small).
   *
   *  Wraps an ordinary RDD to be put on the GPU for OpenCL computations. Call
   *  `cacheGPU` on it to make it persistent on the GPU, otherwise it is
   *  streamed as needed. Be cautious with RDDs that have large partition
   *  objects. These are sent to the executors again every time computations
   *  happen, even if the results are already cached!
   *
   *  The same problem occurs in plain Spark, citing from
   *  ParallelCollectionRDD.scala:
   *  // TODO: Right now, each split sends along its full data, even if later down the RDD chain it gets
   *  // cached. It might be worthwhile to write the data to a file in the DFS and read it in the split
   *  // instead.
   *  // UPDATE: A parallel collection can be checkpointed to HDFS, which achieves this goal.
   *
   *  If you do not care about resilience, you can just use local checkpointing:
   *  val rdd = sc.parallelize(collection).localCheckpoint
   *  rdd.foreach(_ => {}); // force rdd, will be cached after here in spark, needed!
   *  val crdd = CLRDD.wrap(rdd)
   */
  def wrap[T : ClassTag : CLType](wrapped: RDD[T], expectedPartitionSize: Option[Long] = None, cpu: Boolean = OpenCL.CPU) = {
    val elementSize = implicitly[CLType[T]].sizeOf
    var chunkSize = expectedPartitionSize.map(_*elementSize).getOrElse(256*1024*1024L)
    while(chunkSize > Int.MaxValue) {
      chunkSize = (chunkSize + 1)/2
    }
    if(cpu)
      log.info("wrapping for cpu use")
    else
      log.info("wrapping for gpu use")
    val partitionsRDD = new CLWrapPartitionRDD(wrapped, chunkSize.toInt, cpu)
    new CLRDD[T](partitionsRDD, wrapped, cpu)
  }
}

/** An RDD that handles binary data on an OpenCL device.
 *
 *  Methods assume exactly one value per partition inside wrapped!
 */
class CLRDD[T : ClassTag : CLType](val wrapped: RDD[CLPartition[T]], val parentRDD: RDD[_], val cpu: Boolean)
  extends RDD[T](wrapped)
{
  override def compute(split: Partition, ctx: TaskContext) : Iterator[T] = {
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

  def to[B : CLType : ClassTag] : CLRDD[B] = {
    map[B]("return x;")
  }

  /** Apply a simple transformation to this RDD
   *
   *  @param functionBody will be put inside an OpenCL function of the kind
   *    inline B __some_generated_function_name(T x) {
   *      $functionBody
   *    }
   */
  def map[B : CLType : ClassTag](functionBody: String) : CLRDD[B] = {
    new CLRDD(wrapped.map(_.map[B](functionBody)), parentRDD, cpu)
  }

  /** Compute uniformly weighted moving average
   *
   *  Scary looking types enable CLRDD[(Int,Int)].movingAverage -> CLRDD[(Double,Double)]
   */
  def movingAverage(width: Int)(implicit clT: CLType[T]) : CLRDD[clT.doubleCLInstance.elemType] = {
    val clRes = clT.doubleCLInstance
    sliding[clT.doubleCLInstance.elemType](width, 1,
      s"""${clRes.clName} res = ${clRes.zeroName};
      for(int i=0; i<$width; ++i)
        res += convert_${clRes.clName}(GET(i));
      return res/$width;"""
    )(clT.doubleCLInstance.selfInstance, clT.doubleCLInstance.elemClassTag)
  }

  /** Apply a sliding window transformation
   *
   *  The accesible window of size $width will be moved in $stride steps over all data.
   *  Will be horribly inefficient if there are any chunks with less than width-1 elements!
   *  @param functionBody will be put inside an OpenCL function, see map. Input
   *    is available via a GET(i) macro. Valid values for i are in [0;width).
   */
  def sliding[B: CLType : ClassTag](width: Int, stride: Int, functionBody: String) : CLRDD[B] = {
    case class FastPathImpossible(smallSize: Long) extends Exception
    val numPartitions = wrapped.partitions.size
    try{
      val partitionFringes = wrapped.mapPartitionsWithIndex({case (i: Int, it: Iterator[CLPartition[T]]) => {
        val partition = it.next
        it.hasNext
        if(i == 0) { // (part to send to the preceding partition, partition size)
          Iterator(Success((Array[Byte](), partition.count)))
        } else {
          val (session, chunks) = partition.get
          val chunk = chunks.next
          try { 
            val count = chunk.size + chunks.map(c => try {
              c.size
            } finally {
              if(!partition.doCache)
                c.close
            }).sum
            val isLast = !chunks.hasNext && i == numPartitions - 1
            Iterator(
              if(!isLast && chunk.size < width-1) {
                Failure(new FastPathImpossible(chunk.size))
              }
              else
                Success((
                  session.bytesFromChunk(chunk, Math.min(width.toLong-1, chunk.size).toInt, !partition.doCache),
                  count
                ))
              )
          } finally {
            if(!partition.doCache)
              chunk.close
          }
        }
      }}).collect
      //next line can "rethrow" FastPathImpossible from partitionFringes
      val offsets = partitionFringes.map(_.get._2).scan(0L)(_+_) // prefix sums of counts
        .map(preCount => (stride - (preCount % stride)).toInt % stride)
      new CLRDD(wrapped.mapPartitionsWithIndex[CLPartition[B]]({ case (i: Int, it: Iterator[CLPartition[T]]) => {
        val fringe = partitionFringes((i+1)%partitionFringes.size).get._1
        val offset = offsets(i)
        val res = Iterator(new CLSlidingPartition[T, B](functionBody, width, stride, offset, fringe, it.next))
        it.hasNext
        res
      }}), parentRDD, cpu)
    } catch {
      case FastPathImpossible(smallSize) => {
        log.warn("sliding window computation stumbled over a chunk where chunk.size ({}) < width-1 ({}), collecting whole rdd on driver!", smallSize, width-1)
        val count = this.count()
        val newPartitionCount = Math.max(Math.min(numPartitions, count/width/2), 1).toInt
        log.warn("chose to use {} instead of {} partitions", newPartitionCount, numPartitions)
        CLRDD.wrap(context.parallelize(this.collect, newPartitionCount), (count + newPartitionCount - 1)/newPartitionCount, cpu).sliding[B](width, stride, functionBody)
      }
    }
  }

  override def count() : Long = {
    wrapped.map(_.count).fold(0L)(_+_)
  }

  /** Run an associative, commutative reduction. kernel is executed on each
   *  partition, Spark reduce with e, combine afterwards.
   */
  def reduce[B: ClassTag : CLType](kernel: MapReduceKernel[T, B], e: B, combine: (B,B) => B) : B = {
    wrapped.map(_.reduce(kernel, e, combine)).reduce(combine)
  }

  def stats()(implicit evidence: T => Double) : Stats = {
    val clT = implicitly[CLType[T]]
    val clS = implicitly[CLType[Stats]]
    reduce[Stats](
      MapReduceKernel(
        MapFunction[T, Stats](
          Stats.clFromValue,
          clT, clS),
        Stats.clMerge,
        Stats.clZero,
        cpu,
        clT, clS),
      Stats(),
      (x: Stats, y: Stats) => x.merge(y)
    )
  }
  
  def sum(implicit num: Numeric[T]) : T = {
    val clT = implicitly[CLType[T]]
    reduce(MapReduceKernel(
      MapKernel.identity[T],
      "return x+y;",
      clT.zeroName,
      cpu,
      clT, clT
    ), num.zero, ((x: T, y: T) => num.plus(x,y)))
  }

  /**
   * Try to keep the contained data on the device afterwards.
   *
   * Cache state is currently not preserved across failures.
   * This enables computations which need more memory than available to
   * succeed. On the other hand it goes against the intent of the user.
   */
  def cacheGPU = {
    wrapped.cache
    wrapped.foreach(_.cache)
    this
  }

  def uncacheGPU = {
    wrapped.foreach(_.uncache)
    wrapped.unpersist(true)
    this
  }
}

class CLWrapPartitionRDD[T : CLType](val parentRDD: RDD[T], val chunkSize: Int, val cpu: Boolean)
  extends RDD[CLPartition[T]](parentRDD)
{
   override def compute(split: Partition, context: TaskContext) = {
     Iterator(new CLWrapPartition[T](split, parentRDD, chunkSize, OpenCL.get(cpu), cpu))
   }
   override def getPartitions : Array[Partition] = {
     parentRDD.partitions
   }
}

class CLWrapPartition[T : CLType] (val parentPartition: Partition, val parentRDD: RDD[T], val chunkSize: Int, @transient val initialSession: OpenCLSession, val cpu: Boolean)
  extends CLPartition[T] with Serializable
{
  override def src : (OpenCLSession, Iterator[OpenCL.Chunk[T]]) = {
    val session = if(initialSession == null)
      OpenCL.get(cpu)
    else
      initialSession
    val ctx = TaskContext.get
    (session, session.stream(parentRDD.iterator(parentPartition, ctx), chunkSize))
  }
}

trait CLPartition[T] {
  @transient private lazy val log = LoggerFactory.getLogger(getClass)
  @transient protected var session : OpenCLSession = null
  @transient protected var storage : Array[OpenCL.Chunk[T]] = null
  private var _doCache = false
  def doCache = _doCache
  /** Abstract method that "defines" a CLPartition.
   *
   *  Users of CLPartition call get, which potentially caches data.
   */
  protected def src : (OpenCLSession, Iterator[OpenCL.Chunk[T]])
  def get : (OpenCLSession, Iterator[OpenCL.Chunk[T]]) = {
    if(doCache) {
      if(storage == null) {
        val res = src
        session = res._1
        storage = res._2.toArray // TODO potentially leaking Chunks on error
      }
      (session, storage.iterator)
    } else {
      src
    }
  }
  def cache = {
    _doCache = true
  }
  def uncache = {
    _doCache = false
    if(storage != null) {
      storage.foreach(_.close)
      storage = null
      session = null
    }
  }
  def reduce[B](kernel: MapReduceKernel[T, B], e: B, combine: (B,B) => B)
      (implicit clT: CLType[T], clB: CLType[B]) : B = {
    log.trace("reducing {} with {}", Array(this, kernel))
    val (session, chunks) = get
    val future = chunks.map(chunk => {
      try {
        session.reduceChunk(chunk, kernel)
      } finally {
        if(!doCache) chunk.close
      }
    }).foldLeft(Future.successful(e))({ (l: Future[B], r: Future[B]) =>
      val lval = Await.result(l, Duration.Inf)
      r.map(combine(lval,_))
    })
    Await.result(future, Duration.Inf)
  }

  def map[B](functionBody: String)(implicit clT: CLType[T], clB: CLType[B]) : CLPartition[B] = {
    new CLMapPartition[T, B](functionBody, this)
  }
  def count : Long = {
    val (session, chunks) = get
    chunks.map(chunk => {
      val res = chunk.size
      if(!doCache) chunk.close
      res
    }).sum
  }
  def iterator (ctx: TaskContext)(implicit clT: CLType[T]) : Iterator[T] = {
    val (session, chunks) = get
    chunks.flatMap(chunk => {
      var res = session.ChunkIterator(chunk)
      if(!doCache)
        chunk.close
      res
    })
  }
}

class CLMapPartition[A, B](val functionBody: String, val parent: CLPartition[A])(implicit clA: CLType[A], clB: CLType[B]) 
  extends CLPartition[B]
{
  val f = MapFunction[A,B](functionBody, clA, clB)
  def composed[C: CLType](g: MapKernel[B,C]) : (OpenCLSession, Iterator[OpenCL.Chunk[C]]) = {
    val fg = f.compose(g)
    (doCache, parent) match {
      case (false, p : CLMapPartition[_, A]) => p.composed(fg)
      case _ =>  {
        val (session, parentChunks) = parent.get
        val mappedChunks = parentChunks.map(c => {
          if(parent.doCache) {
            session.mapChunk[A,C](c, fg, false)
          } else {
            session.mapChunk[A,C](c, fg, true)
          }
        })
        (session, mappedChunks)
      }
    }
  }
  override def src = {
    composed(MapKernel.identity[B])
  }
  override def reduce[C](kernel: MapReduceKernel[B, C], e: C, combine: (C,C) => C)
      (implicit clB: CLType[B], clC: CLType[C]) : C = {
    if(doCache)
      super.reduce(kernel, e, combine)
    else
      parent.reduce(kernel.precomposeMap(f), e, combine)
  }
}

class CLSlidingPartition[A, B](val functionBody: String, val width: Int, val stride: Int, val offset: Int, val fringe: Array[Byte], val parent: CLPartition[A])(implicit clA: CLType[A], clB: CLType[B])
  extends CLPartition[B]
{
  @transient private lazy val log = LoggerFactory.getLogger(getClass)
  val f = WindowReduction[A, B](functionBody, clA, clB)
  override def src = {
    val (session, parentChunks : Iterator[OpenCL.Chunk[A]]) = parent.get
    val fringeChunk : OpenCL.Chunk[A] = session.chunkFromBytes[A](fringe)
    val (it, nit) = parentChunks.duplicate
    nit.next()
    val zipped = it.zip(nit ++ Iterator(fringeChunk))
    val mappedChunks = zipped.map({case (c, nc) => {
      try{
        session.mapSliding(c, nc, f, width, stride, offset)
        } finally {
          if(nc eq fringeChunk)
            nc.close
          if(!parent.doCache) {
            c.close
          }
        }
    }})
    (session, mappedChunks)
  }
}
