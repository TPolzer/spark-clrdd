package de.qaware.chronix.cl.benchmarks

import org.jocl.CL._
import de.qaware.chronix.cl._
import org.apache.spark._
import org.apache.spark.rdd._
import org.openjdk.jmh.annotations._
import org.slf4j.LoggerFactory
import java.util.concurrent.TimeUnit
import org.apache.spark.mllib.rdd.RDDFunctions._
import scala.concurrent.duration._
import scala.concurrent.Await

object BenchmarksCommon {
  def arraySum(a: Array[Double]) = {
    var res = 0.0
    var i = 0
    while(i != a.length) {
      res += a(i)
      i += 1
    }
    res
  }

  def waitAndClose(c: Iterator[OpenCL.Chunk[_]]) = {
    val chunks = c.toArray
    clWaitForEvents(chunks.size, chunks.map(_.ready))
    chunks.foreach(_.close)
  }
}

@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 10, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 4, time = 2, timeUnit = TimeUnit.SECONDS)
@Timeout(time = 1, timeUnit = TimeUnit.MINUTES)
@Fork(value = 1, jvmArgsAppend = Array("-Xmx10g"))
abstract class BenchmarksCommon {
  private val log = LoggerFactory.getLogger(getClass)
  lazy val sc = {
    val sc = SparkContext.getOrCreate()
    sc.setLogLevel("WARN")
    sc
  }
  @TearDown
  def close = {
    sc.stop
  }

  @Setup
  def setExecutors : Unit = {
    val props = System.getProperties();
    //ensure even distribution of data over devices
    props.setProperty("spark.executor.cores", execPerNode.toString)
    props.setProperty("spark.cores.max", partitions.toString)
  }

  def rdd: RDD[Double]
  def crdd: CLRDD[Double]
  var cpu: Boolean
  def totalSize: Long
  var size: Long
  def execPerNode : Int = {
    if(cpu)
      4 * 2 // cores * oversubscription
    else
      1 * 2 // GPUs * oversubscription
  }
  def partitions : Int = execPerNode * 1 // * numberOfNodes
  
  lazy val chunks = {
    val chunkSize = size*1024*1024/execPerNode
    assert(chunkSize <= Int.MaxValue)
    for(i <- 1 to execPerNode) yield {
      val session = OpenCL.get(cpu)
      val it = session.stream((1L to chunkSize/8).iterator.map(_.toDouble), chunkSize.toInt)
      val res = it.next
      assert(!it.hasNext)
      (session, res)
    }
  }

  def sizeLoop[A](f: => A) {
    var i = 0L;
    while(i != totalSize) {
      assert(i < totalSize)
      f
      i += size
    }
  }

  def clsum = { 
    var res = crdd.to[Double].sum
    var i = size;
    while(i != totalSize) {
      assert(i < totalSize)
      res += crdd.to[Double].sum
      i += size
    }   
    res 
  }

  def rawclsum = {
    var res = 0.0
    val clL = implicitly[CLType[Double]]
    val clD = implicitly[CLType[Double]]
    val kernel = MapReduceKernel(
      MapFunction("return x;", clL, clD),
      "return x+y;",
      "0.0",
      cpu,
      clL, clD
    ) 
    sizeLoop {
      val fs = chunks.par.map({case (session: OpenCLSession, chunk:OpenCL.Chunk[Double]) => {
        val f = session.reduceChunk(chunk, kernel)
        Await.result(f, Duration.Inf)
      }})
      res += fs.sum
    }
    res
  }

  def clstats = {
    var res = crdd.stats
    var i = size;
    while(i != totalSize) {
      assert(i < totalSize)
      res = crdd.stats.merge(res)
      i += size
    }   
    assert(res.count == totalSize.toLong*1024*1024/8, res.count)
    res 
  }

  def clMovAvg(windowSize: Int) = sizeLoop {
    crdd.movingAverage(windowSize).wrapped.foreach(p => BenchmarksCommon.waitAndClose(p.get._2))
  }

  def movAvg(windowSize: Int) = sizeLoop {
    rdd.sliding(windowSize).map(a => BenchmarksCommon.arraySum(a)/a.length).foreach(_ => {})
  }

  def clChkAvg(chunkSize: Int) = sizeLoop {
    crdd.sliding[Double](chunkSize, chunkSize, s"""
      double res = 0;
      for(int i = 0; i < $chunkSize; ++i)
        res += GET(i);
      return res/$chunkSize;
      """).wrapped.foreach(p => BenchmarksCommon.waitAndClose(p.get._2))
  }

  def chkAvg(chunkSize: Int) = sizeLoop {
    rdd.sliding(chunkSize, chunkSize).map(a => BenchmarksCommon.arraySum(a)/a.length).foreach(_ => {})
  }
  
  def stats = {
    var res = rdd.stats
    var i = size;
    while(i != totalSize) {
      assert(i < totalSize)
      res.merge(rdd.stats)
      i += size
    }
    assert(res.count == totalSize.toLong*1024*1024/8, res.count)
    res
  }

  def fastSum = {
    var res = 0.0
    sizeLoop {
      res += sc.runJob(rdd, ((it:Iterator[Double]) => {
        var total = 0.0;
        while(it.hasNext) {
          total += it.next
        }
        total
      })).sum
    }
    res
  }
}
