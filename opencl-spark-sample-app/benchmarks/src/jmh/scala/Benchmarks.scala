package de.qaware.chronix.cl.benchmarks

import de.qaware.chronix.cl._
import org.apache.spark._
import org.openjdk.jmh.annotations._
import org.slf4j.LoggerFactory
import java.util.concurrent.TimeUnit

object Benchmarks {
  final val size = 1024
}

@State(Scope.Benchmark)
@Warmup(iterations = 15, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 1, timeUnit = TimeUnit.SECONDS)
@Timeout(time = 1, timeUnit = TimeUnit.MINUTES)
@OperationsPerInvocation(Benchmarks.size)
@Fork(value = 1, jvmArgsAppend = Array("-Xmx8g"))
class Benchmarks {
  val sc = new SparkContext("local[*]", "chronix cl benchmark")
  sc.setLogLevel("WARN")

  //@Param(Array("4", "16", "64", "128", "512", "1024", "2048"))
  @Param(Array("16", "128", "1024"))
  private var size = 128L
  //@Param(Array("1", "4", "8", "16", "32"))
  @Param(Array("1", "4", "8", "16"))
  private var partitions = 1
  @Param(Array("true", "false"))
  private var cpu = true


  private val log = LoggerFactory.getLogger(getClass)

  lazy val rdd = {
    OpenCL.CPU = cpu
    sc.range(0, size*1024*1024/8, 1, partitions).cache
  }
  lazy val crdd = CLRDD.wrap(rdd).cacheGPU

  @Benchmark
  def clsum = {
    var res = crdd.to[Double].sum
    var i = size;
    while(i != Benchmarks.size) {
      assert(i < Benchmarks.size)
      res += crdd.to[Double].sum
      i += size
    }
    res
  }

  @Benchmark
  def clstats = {
    var res = crdd.stats
    var i = size;
    while(i != Benchmarks.size) {
      assert(i < Benchmarks.size)
      res = crdd.stats.merge(res)
      i += size
    }
    assert(res.count == Benchmarks.size.toLong*1024*1024/8, res.count)
    res
  }
  
  @Benchmark
  def sum = {
    assert(cpu)
    var res = rdd.sum
    var i = size;
    while(i != Benchmarks.size) {
      assert(i < Benchmarks.size)
      res += rdd.sum
      i += size
    }
    res
  }

  @Benchmark
  def stats = {
    assert(cpu)
    var res = rdd.stats
    var i = size;
    while(i != Benchmarks.size) {
      assert(i < Benchmarks.size)
      res.merge(rdd.stats)
      i += size
    }
    assert(res.count == Benchmarks.size.toLong*1024*1024/8, res.count)
    res
  }
}
