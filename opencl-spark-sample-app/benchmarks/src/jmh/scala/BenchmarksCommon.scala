package de.qaware.chronix.cl.benchmarks

import org.jocl.CL._
import de.qaware.chronix.cl._
import org.apache.spark._
import org.apache.spark.rdd._
import org.openjdk.jmh.annotations._
import org.slf4j.LoggerFactory
import java.util.concurrent.TimeUnit
import org.apache.spark.mllib.rdd.RDDFunctions._

object BenchmarksCommon {
  def arraySum(a: Array[Long]) = {
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
@Warmup(iterations = 13, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 8, time = 1, timeUnit = TimeUnit.SECONDS)
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

  def rdd: RDD[Long]
  def crdd: CLRDD[Long]
  def totalSize: Long
  var size: Long

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
}
