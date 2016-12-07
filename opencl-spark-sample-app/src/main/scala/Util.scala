trait HashcodeCaching { self: Product =>
  override lazy val hashCode: Int = {
    scala.runtime.ScalaRunTime._hashCode(this)
  }
}
