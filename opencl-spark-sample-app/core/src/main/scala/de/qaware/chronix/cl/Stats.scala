/*
 * Heavily based on Apache Spark's StatCounter.scala, its license header is
 * reproduced below. It cites Welford and Chan's
 * [[http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 * algorithms]] for running variance.
 *
 * Interoperability is sadly impossible, since the original class is closed
 * against extension.
 */
/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package de.qaware.chronix.cl

import scala.reflect.ClassTag
import java.nio.ByteBuffer

case class Stats(
  n: Long = 0,     // Running count of our values
  mu: Double = 0,  // Running mean of our values
  m2: Double = 0,  // Running variance numerator (sum of (x - mean)^2)
  maxValue: Double = Double.NegativeInfinity, // Running max of our values
  minValue: Double = Double.PositiveInfinity // Running min of our values
) {
  def count: Long = n

  def mean: Double = mu

  def sum: Double = n * mu

  def max: Double = maxValue

  def min: Double = minValue

  def merge(other: Stats): Stats = {
    if (n == 0) {
      other
    } else if (other.n != 0) {
      val delta = other.mu - this.mu
      var mu = 0.0
      if (other.n * 10 < this.n) {
        mu = this.mu + (delta * other.n) / (this.n + other.n)
      } else if (this.n * 10 < other.n) {
        mu = other.mu - (delta * this.n) / (this.n + other.n)
      } else {
        mu = (this.mu * this.n + other.mu * other.n) / (this.n + other.n)
      }
      val m2 = this.m2 + other.m2 + (delta * delta * this.n * other.n) / (this.n + other.n)
      val n = this.n + other.n
      val maxValue = math.max(this.maxValue, other.maxValue)
      val minValue = math.min(this.minValue, other.minValue)
      Stats(n, mu, m2, maxValue, minValue)
    } else {
      this
    }
  }

  def variance: Double = {
	if (n == 0) {
	  Double.NaN
	} else {
	  m2 / n
	}
  }

  def sampleVariance: Double = {
    if (n <= 1) {
      Double.NaN
    } else {
      m2 / (n - 1)
    }
  }

  def stdev: Double = math.sqrt(variance)

  def sampleStdev: Double = math.sqrt(sampleVariance)

  override def toString: String = {
    "(count: %d, mean: %f, stdev: %f, max: %f, min: %f)".format(count, mean, stdev, max, min)
  }
}

object Stats {
  def clFromValue : String = """
    double value = x;
    double n = 1;
    double mu = value / n;
    double m2 = value * (value - mu);
    double maxValue = value;
    double minValue = value;
    double _;
    return (double8)(n, mu, m2, maxValue, minValue, _, _, _);
  """
  def clZero : String = """
    (double8)(0, 0, 0, -INFINITY, INFINITY, 0, 0, 0)
  """
  def clMerge : String = """
    if (x.s0 == 0) {
      return y;
    } else if (y.s0 != 0) {
      double delta = y.s1 - x.s1;
      double s1 = 0.0;
      if (y.s0 * 10 < x.s0) {
        s1 = x.s1 + (delta * y.s0) / (x.s0 + y.s0);
      } else if (x.s0 * 10 < y.s0) {
        s1 = y.s1 - (delta * x.s0) / (x.s0 + y.s0);
      } else {
        s1 = (x.s1 * x.s0 + y.s1 * y.s0) / (x.s0 + y.s0);
      }
      double s2 = x.s2 + y.s2 + (delta * delta * x.s0 * y.s0) / (x.s0 + y.s0);
      double s0 = x.s0 + y.s0;
      double s3 = max(x.s3, y.s3);
      double s4 = min(x.s4, y.s4);
      double _;
      return (double8)(s0, s1, s2, s3, s4, _, _, _);
    } else {
      return x;
    }
  """
  trait CLStats extends CLType[Stats] {
    val cld = CLDouble8
    override val clName = cld.clName
    override val sizeOf = cld.sizeOf
    override val header = cld.header
    override val zeroName = cld.zeroName
    override def fromByteBuffer(idx: Int, rawBuffer: ByteBuffer) : Stats = {
      val d = cld.fromByteBuffer(idx, rawBuffer)
      new Stats(d._1.toLong, d._2, d._3, d._4, d._5)
    }
    override def toByteBuffer(idx: Int, rawBuffer: ByteBuffer, v: Stats) = {
      cld.toByteBuffer(idx, rawBuffer, (v.n.toDouble, v.mu, v.m2, v.maxValue, v.minValue, 0.0, 0.0, 0.0))
    }
    override type doubleCLType = CLStats
    override val doubleCLInstance = CLStats
    override val selfInstance = CLStats
    override val elemClassTag = implicitly[ClassTag[Stats]]
  }
  implicit object CLStats extends CLStats
}
