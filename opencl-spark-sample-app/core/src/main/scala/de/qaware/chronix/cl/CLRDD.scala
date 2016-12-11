package de.qaware.chronix.cl

import org.apache.spark.rdd._
import org.apache.spark._
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._

import scala.reflect.ClassTag

import scala.concurrent.duration._
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.ExecutionContext.Implicits.global

object CLRDD
{
  @transient lazy private val sc = SparkContext.getOrCreate
  def wrap[T : ClassTag : CLType](wrapped: RDD[T], expectedPartitionSize: Option[Long] = None) = {
    val partitions = sc.broadcast(wrapped.partitions)
    val elementSize = implicitly[CLType[T]].sizeOf
    var chunkSize = expectedPartitionSize.map(_*elementSize).getOrElse(256*1024*1024L)
    while(chunkSize > Int.MaxValue) {
      chunkSize = (chunkSize + 1)/2
    }
    val partitionsRDD = wrapped.mapPartitionsWithIndex( { case (idx: Int, _) =>
        Iterator(new CLWrapPartition[T](partitions.value(idx), wrapped, chunkSize.toInt).asInstanceOf[CLPartition[T]]) } )
    new CLRDD[T](partitionsRDD, wrapped)
  }
}

class CLRDD[T : ClassTag : CLType](val wrapped: RDD[CLPartition[T]], val parentRDD: RDD[_])
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
    new CLRDD(wrapped.map(_.map[B]("return x;")), parentRDD)
  }

  def map[B : CLType : ClassTag](functionBody: String) : CLRDD[B] = {
    new CLRDD(wrapped.map(_.map[B](functionBody)), parentRDD)
  }

  override def count : Long = {
    wrapped.map(_.count).fold(0L)(_+_)
  }
  
  def sum(implicit num: Numeric[T]) : T = {
    wrapped.map(_.sum).reduce(num.plus(_, _))
  }

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

class CLWrapPartition[T : CLType] (val parentPartition: Partition, val parentRDD: RDD[T], val chunkSize: Int)
  extends CLPartition[T]
{
  override def src : (OpenCLSession, Iterator[OpenCL.Chunk[T]]) = {
    val ctx = TaskContext.get
    val session = OpenCL.get
    (session, session.stream(parentRDD.iterator(parentPartition, ctx), chunkSize))
  }
}

trait CLPartition[T] {
  @transient private lazy val log = LoggerFactory.getLogger(getClass)
  @transient protected var session : OpenCLSession = null
  @transient protected var storage : Array[OpenCL.Chunk[T]] = null
  private var _doCache = false
  def doCache = _doCache
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
    if(storage != null) {
      _doCache = false
      storage.foreach(_.close)
      storage = null
      session = null
    }
  }
  def reduce[B](kernel: MapReduceKernel[T, B], e: B, combine: (B,B) => B)
      (implicit clT: CLType[T], clB: CLType[B]) : B = {
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
  def sum(implicit clT: CLType[T], num: Numeric[T]) : T = reduce(MapReduceKernel(
    MapKernel.identity[T],
    "return x+y;",
    clT.zeroName,
    OpenCL.CPU,
    clT, clT
  ), num.zero, ((x: T, y: T) => num.plus(x,y)))

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
