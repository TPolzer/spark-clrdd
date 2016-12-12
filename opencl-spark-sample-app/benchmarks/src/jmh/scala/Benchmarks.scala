package de.qaware.chronix.cl.benchmarks

import de.qaware.chronix.cl._
import org.apache.spark._
import org.openjdk.jmh.annotations._
import org.slf4j.LoggerFactory
import java.util.concurrent.TimeUnit

@State(Scope.Benchmark)
@Warmup(iterations = 15, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 15, time = 1, timeUnit = TimeUnit.SECONDS)
@OperationsPerInvocation(10*1024)
@Fork(value = 2, jvmArgsAppend = Array("-Xmx8g"))
class Benchmarks {
  val sc = new SparkContext("local[*]", "chronix cl benchmark")
  sc.setLogLevel("WARN")

  @Param(Array("4", "16", "64", "128", "512", "1024", "2048", "5120"))
  private var size = 128L
  @Param(Array("1", "4", "8", "16", "32"))
  private var partitions = 1

  private val log = LoggerFactory.getLogger(getClass)

  lazy val rdd = {
    sc.range(0, size*1024*1024/8, 1, partitions).cache
  }
  lazy val crdd = CLRDD.wrap(rdd).cacheGPU

  @Benchmark
  def clsum = {
    var res = crdd.to[Double].sum
    var i = size;
    while(i != 10*1024) {
      assert(i < 10*1024)
      res += crdd.to[Double].sum
      i += size
    }
    res
  }

  @Benchmark
  def clstats = {
    var res = crdd.stats
    var i = size;
    while(i != 10*1024) {
      assert(i < 10*1024)
      res = crdd.stats.merge(res)
      i += size
    }
    assert(res.count == 10L*1024*1024*1024/8, res.count)
    res
  }
  
  @Benchmark
  def sum = {
    var res = rdd.sum
    var i = size;
    while(i != 10*1024) {
      assert(i < 10*1024)
      res += rdd.sum
      i += size
    }
    res
  }

  @Benchmark
  def stats = {
    var res = rdd.stats
    var i = size;
    while(i != 10*1024) {
      assert(i < 10*1024)
      res.merge(rdd.stats)
      i += size
    }
    assert(res.count == 10*1024, res.count)
    res
  }
}
