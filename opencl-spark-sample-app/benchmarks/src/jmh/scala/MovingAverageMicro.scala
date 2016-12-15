package de.qaware.chronix.cl.benchmarks

import de.qaware.chronix.cl._
import org.apache.spark._
import org.openjdk.jmh.annotations._
import org.slf4j.LoggerFactory
import java.util.concurrent.TimeUnit
import org.jocl.CL._

object AverageMicro {
  final val size = 1024
}

@State(Scope.Benchmark)
@Warmup(iterations = 15, time = 1, timeUnit = TimeUnit.SECONDS)
@Measurement(iterations = 10, time = 1, timeUnit = TimeUnit.SECONDS)
@Timeout(time = 10, timeUnit = TimeUnit.SECONDS)
@OperationsPerInvocation(AverageMicro.size)
@Fork(value = 1, jvmArgsAppend = Array("-Xmx8g"))
class AverageMicro {
  lazy val sc = {
    val sc = new SparkContext("local[*]", "chronix cl benchmark")
    sc.setLogLevel("WARN")
    sc
  }

  @Param(Array("64", "1024"))
  private var size : Long = 0
  @Param(Array("2", "8"))
  private var partitions : Int = 0
  @Param(Array("1","10","100"))
  private var windowSize : Int = 0

  private val log = LoggerFactory.getLogger(getClass)

  lazy val rdd = sc.range(0, size*1024*1024/8, 1, partitions)
  lazy val crdd = CLRDD.wrap(rdd, Some((size*1024*1024/8+partitions-1)/partitions)).cacheGPU

  @Benchmark
  def clMovAvg = {
    var i = 0L;
    while(i != AverageMicro.size) {
      assert(i < AverageMicro.size)
      val res = crdd.movingAverage(windowSize)
      res.wrapped.foreach(p => {
        val (session, chunks) = p.get
        val forcedChunks = chunks.toArray
        clWaitForEvents(forcedChunks.size, forcedChunks.map(_.ready))
        forcedChunks.foreach(_.close)
      })
      i += size
    }
  }
}
