package de.qaware.chronix.cl.benchmarks

import de.qaware.chronix.cl._
import org.jocl._
import org.jocl.CL._
import org.apache.spark._
import org.openjdk.jmh.annotations._
import org.slf4j.LoggerFactory
import java.util.concurrent.TimeUnit

object BenchmarksLarge {
//  final val size = 65536
  final val size = 2048
}

@OperationsPerInvocation(BenchmarksLarge.size)
class BenchmarksLarge extends BenchmarksCommon {

  @Param(Array("64", "128", "512", "1024"))
//  @Param(Array("4096", "16384", "32768", "65536"))
  override var size : Long = 0
  lazy val partitions : Int = {
    if(cpu)
//      16*5
      8
    else
//      4*5 * 2
      2
  }
  @Param(Array("true", "false"))
  var cpu = true

  override val totalSize = BenchmarksLarge.size.toLong


  private val log = LoggerFactory.getLogger(getClass)

  override lazy val rdd = {
    OpenCL.CPU = cpu
    sc.range(0, size*1024*1024/8, 1, partitions).cache
  }
  override lazy val crdd = {
    OpenCL.CPU = cpu
    var chunks = partitions
    log.warn("rdd has {} partitions", chunks)
    var crdd : CLRDD[Long] = null
    while(crdd == null) {
      try { 
        crdd = CLRDD.wrap(sc.range(0, size*1024*1024/8, 1, partitions), Some((size*1024*1024/8+chunks-1)/chunks)).cacheGPU
        crdd.sum
      } catch {
        case (e: Throwable) => {
          crdd.uncacheGPU
          crdd = null
          /*
           * some GPUs dont like BIG allocations, so try smaller
           */
          if(size/chunks < 128) {
            throw e
          } else {
            chunks *= 2
          }
        }
      }
    }
    log.warn("crdd has {} chunks", chunks)
    crdd
  }

  @Benchmark
  override def clsum = super.clsum

  @Benchmark
  override def clstats = super.clstats

  @Benchmark
  override def stats = {
    assert(cpu)
    super.stats
  }

  @Benchmark
  def clMovAvg32 = super.clMovAvg(32)
  
  @Benchmark
  def clMovAvg16 = super.clMovAvg(16)
  
  @Benchmark
  def clMovAvg4 = super.clMovAvg(4)

  @Benchmark
  def clChkAvg16 = super.clChkAvg(16)
  
  @Benchmark
  def clChkAvg128 = super.clChkAvg(128)
  
  @Benchmark
  def clChkAvg1024 = super.clChkAvg(1024)

  @Benchmark
  def chkAvg16 = {
    assert(cpu)
    super.chkAvg(16)
  }
  
  @Benchmark
  def chkAvg128 = {
    assert(cpu)
    super.chkAvg(128)
  }
  
  @Benchmark
  def chkAvg1024 = {
    assert(cpu)
    super.chkAvg(1024)
  }
}

object BenchmarksSmall {
  final val size = 64
}

@OperationsPerInvocation(BenchmarksSmall.size)
class BenchmarksSmall extends BenchmarksCommon {

  @Param(Array("1", "2", "4", "16", "32"))
  override var size : Long = 0
  lazy val partitions : Int = {
    if(cpu)
//      16*5
      8
    else
//      4*5
      2
  }
  @Param(Array("true", "false"))
  var cpu = true

  override val totalSize = BenchmarksSmall.size.toLong


  private val log = LoggerFactory.getLogger(getClass)

  override lazy val rdd = {
    OpenCL.CPU = cpu
    sc.range(0, size*1024*1024/8, 1, partitions).cache
  }
  override lazy val crdd = {
    OpenCL.CPU = cpu
    CLRDD.wrap(rdd, Some((size*1024*1024/8+partitions-1)/partitions)).cacheGPU
  }

  @Benchmark
  override def clsum = super.clsum

  @Benchmark
  override def clstats = super.clstats

  @Benchmark
  override def stats = {
    assert(cpu)
    super.stats
  }
 
  @Benchmark
  def clMovAvg128 = super.clMovAvg(128)

  @Benchmark
  def clMovAvg32 = super.clMovAvg(32)
  
  @Benchmark
  def clMovAvg16 = super.clMovAvg(16)
  
  @Benchmark
  def clMovAvg4 = super.clMovAvg(4)
  
  @Benchmark
  def MovAvg128 = {
    assert(cpu)
    super.movAvg(128)
  }

  @Benchmark
  def MovAvg32 = {
    assert(cpu)
    super.movAvg(32)
  }
  
  @Benchmark
  def MovAvg16 = {
    assert(cpu)
    super.movAvg(16)
  }
  
  @Benchmark
  def MovAvg4 = {
    assert(cpu)
    super.movAvg(4)
  }

  @Benchmark
  def clChkAvg16 = super.clChkAvg(16)
  
  @Benchmark
  def clChkAvg128 = super.clChkAvg(128)
  
  @Benchmark
  def clChkAvg1024 = super.clChkAvg(1024)
  
  @Benchmark
  def chkAvg16 = {
    assert(cpu)
    super.chkAvg(16)
  }
  
  @Benchmark
  def chkAvg128 = {
    assert(cpu)
    super.chkAvg(128)
  }
  
  @Benchmark
  def chkAvg1024 = {
    assert(cpu)
    super.chkAvg(1024)
  }
}
