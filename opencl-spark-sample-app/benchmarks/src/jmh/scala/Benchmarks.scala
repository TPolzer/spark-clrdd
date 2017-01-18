package de.qaware.chronix.cl.benchmarks

import de.qaware.chronix.cl._
import org.jocl._
import org.jocl.CL._
import org.apache.spark._
import org.openjdk.jmh.annotations._
import org.slf4j.LoggerFactory
import java.util.concurrent.TimeUnit

/*
 * Several constants in this file need tuning depending on available (GPU)
 * memory and cluster size. This is less than optimal, but static annotations
 * can only get you so far...
 */

object BenchmarksLarge {
  final val size = 2048
//  final val size = 16384
}

@OperationsPerInvocation(BenchmarksLarge.size)
class BenchmarksLarge extends BenchmarksCommon {

  @Param(Array("64", "128", "512", "1024"))
//  @Param(Array("512", "1024", "4096", "16384"))
  override var size : Long = 0
  @Param(Array("true", "false"))
  override var cpu = true

  override val totalSize = BenchmarksLarge.size.toLong


  private val log = LoggerFactory.getLogger(getClass)

  override lazy val rdd = {
    OpenCL.CPU = cpu
    sc.range(0, size*1024*1024/8, 1, partitions).map(_.toDouble).cache
  }
  override lazy val crdd = {
    OpenCL.CPU = cpu
    var chunks = partitions
    log.warn("rdd has {} partitions", chunks)
    var crdd : CLRDD[Double] = null
    while(crdd == null) {
      try { 
        crdd = CLRDD.wrap(sc.range(0, size*1024*1024/8, 1, partitions).map(_.toDouble), Some((size*1024*1024/8+chunks-1)/chunks)).cacheGPU
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
  override def rawclsum = super.rawclsum

  @Benchmark
  override def clstats = super.clstats

  @Benchmark
  override def stats = {
    assert(cpu)
    super.stats
  }

  @Benchmark
  override def fastSum = {
    assert(cpu)
    super.fastSum
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
//  final val size = 128
  final val size = 32
}

@OperationsPerInvocation(BenchmarksSmall.size)
class BenchmarksSmall extends BenchmarksCommon {

//  @Param(Array("8", "16", "32", "64", "128"))
  @Param(Array("1", "2", "4", "16", "32"))
  override var size : Long = 0
  @Param(Array("true", "false"))
  override var cpu = true

  override val totalSize = BenchmarksSmall.size.toLong


  private val log = LoggerFactory.getLogger(getClass)

  override lazy val rdd = {
    OpenCL.CPU = cpu
    sc.range(0, size*1024*1024/8, 1, partitions).map(_.toDouble).cache
  }
  override lazy val crdd = {
    OpenCL.CPU = cpu
    CLRDD.wrap(rdd, Some((size*1024*1024/8+partitions-1)/partitions)).cacheGPU
  }

  @Benchmark
  override def clsum = super.clsum

  @Benchmark
  override def rawclsum = super.rawclsum

  @Benchmark
  override def clstats = super.clstats

  @Benchmark
  override def stats = {
    assert(cpu)
    super.stats
  }

  @Benchmark
  override def fastSum = {
    assert(cpu)
    super.fastSum
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
