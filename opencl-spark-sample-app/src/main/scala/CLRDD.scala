package de.qaware.chronix

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
  def wrap[T : ClassTag : CLType](wrapped: RDD[T]) = {
    val partitions = sc.broadcast(wrapped.partitions)
    val partitionsRDD = wrapped.mapPartitionsWithIndex( { case (idx: Int, _) =>
        Iterator(new CLWrapPartition[T](partitions.value(idx), wrapped).asInstanceOf[CLPartition[T]]) } )
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
  
  def sum(implicit num: CLNumeric[T]) : T = {
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

class CLWrapPartition[T : CLType] (val parentPartition: Partition, val parentRDD: RDD[T])
  extends CLPartition[T]
{
  override def src : (OpenCLSession, Iterator[OpenCL.Chunk[T]]) = {
    val ctx = TaskContext.get
    val session = OpenCL.get
    (session, session.stream(parentRDD.iterator(parentPartition, ctx)))
  }
}

trait CLPartition[T] { self =>
  @transient private lazy val log = LoggerFactory.getLogger(getClass)
  @transient protected var session : OpenCLSession = null
  @transient protected var storage : Array[OpenCL.Chunk[T]] = null
  @transient protected var doCache = false
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
  def sum(implicit clT: CLNumeric[T]) : T = {
    val (session, chunks) = get
    val future = chunks.map(chunk => {
      try {
        session.reduceChunk[T](chunk, "", clT.zeroName, "return x+y;")
      } finally {
        if(!doCache) chunk.close
      }
    }).foldLeft(Future.successful(clT.zero))({ (l: Future[T], r: Future[T]) =>
      val lval = Await.result(l, Duration.Inf)
      r.map(clT.plus(lval,_))
    })
    Await.result(future, Duration.Inf)
  }
  def map[B](functionBody: String)(implicit clT: CLType[T], clB: CLType[B]) : CLPartition[B] = {
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
