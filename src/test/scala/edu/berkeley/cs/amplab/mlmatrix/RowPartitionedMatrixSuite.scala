package edu.berkeley.cs.amplab.mlmatrix

import org.scalatest.FunSuite
import org.apache.spark.SparkContext
import breeze.linalg._

class RowPartitionedMatrixSuite extends FunSuite with LocalSparkContext with Logging {

  test("reduceRowElements()") {
    sc = new SparkContext("local", "test")
    val testMat = RowPartitionedMatrix.fromArray(
      sc.parallelize(Seq(
        Array[Double](1, 2, 3),
        Array[Double](1, 9, -1),
        Array[Double](0, 0, 1),
        Array[Double](0, 1, 0)
      ), 2), // row-major, laid out as is
      Seq(2, 2),
      3
    )
    val rowProducts = testMat.reduceRowElements(_ * _)

    assert(rowProducts.collect().toArray === Array(6, -9, 0, 0),
      "reduceRowElements() does not return correct answers!")
    assert(rowProducts.numRows() === 4, "reduceRowElements() returns a result with incorrect row count!")
    assert(rowProducts.numCols() === 1, "reduceRowElements() returns a result with incorrect col count!")
  }

  test("reduceColElements() and colSums()") {
    sc = new SparkContext("local", "test")
    val testMat = RowPartitionedMatrix.fromArray(
      sc.parallelize(Seq(
        Array[Double](1, 2, 3),
        Array[Double](1, 9, -1),
        Array[Double](1, 0, 1),
        Array[Double](1618, 1, 4)
      ), 2), // row-major, laid out as is
      Seq(2, 2),
      3
    )
    val colProducts = testMat.reduceColElements(_ * _)

    assert(colProducts.collect().toArray === Array(1618, 0, -12),
      "reduceColElements() does not return correct answers!")
    assert(colProducts.numRows() === 1, "reduceColElements() returns a result with incorrect row count!")
    assert(colProducts.numCols() === 3, "reduceColElements() returns a result with incorrect col count!")

    assert(testMat.colSums() === Seq(1621, 12, 7), "colSums() returns incorrect sums!")
  }

  test("rowSums()") {
    sc = new SparkContext("local", "test")
    val testMat = RowPartitionedMatrix.fromArray(
      sc.parallelize(Seq(
        Array[Double](1, 2, 3),
        Array[Double](1, 9, -1),
        Array[Double](0, 0, 1),
        Array[Double](0, 1, 0)
      ), 4), // row-major, laid out as is
      Seq(1, 1, 1, 1),
      3
    )
    assert(testMat.rowSums() === Seq(6, 9, 1, 1), "rowSums() returns incorrect sums!")
  }

  test("slicing using various apply() methods") {
    sc = new SparkContext("local", "test")
    val testMat = RowPartitionedMatrix.fromArray(
      sc.parallelize(Seq(
        Array[Double](1, 2, 3),
        Array[Double](1, 9, -1),
        Array[Double](0, 1618, 1),
        Array[Double](0, 1, 0)
      ), 4), // row-major, laid out as is
      Seq(1, 1, 1, 1),
      3
    )
    assert(testMat(::, Range(1, 2)).collect().toArray === Array(2, 9, 1618, 1))
    assert(testMat(::, Range(1, 3)).collect().toArray === Array(2, 9, 1618, 1, 3, -1, 1, 0))
    assert(testMat(Range(1, 2), ::).collect().toArray === Array(1, 9, -1))
    assert(testMat(Range(0, 5), ::).collect().toArray === testMat.collect().toArray)
    assert(testMat(Range(2, 3), Range(1, 2)).collect().toArray.head === 1618)
    assert(testMat(Range(2, 3), Range(1, 3)).collect().toArray === Array(1618, 1))

    assert(testMat(Range(2, 2), Range(1, 3)).collect().toArray.isEmpty)
  }

  test("delete(cols, axis)") {
    sc = new SparkContext("local", "test")
    val testMat = RowPartitionedMatrix.fromArray(
      sc.parallelize(Seq(
        Array[Double](1, 2, 3),
        Array[Double](1, 9, -1),
        Array[Double](0, 0, 1),
        Array[Double](0, 1, 0)
      ), 2), // row-major, laid out as is
      Seq(2, 2),
      3
    )

    val testMatTruncated = testMat.delete(Seq(0, 2), Axis._1)

    assert(testMatTruncated.collect().toArray === Array(2, 9, 0, 1),
    "delete(cols, axis) does not return correct answers!")
    assert(testMatTruncated.numRows() === 4, "delete(cols, axis) returns a result with incorrect row count!")
    assert(testMatTruncated.numCols() === 1, "delete(cols, axis) returns a result with incorrect col count!")
  }

  test("horzcat(other)") {
    sc = new SparkContext("local", "test")
    val testMat = RowPartitionedMatrix.fromArray(
      sc.parallelize(Seq(
        Array[Double](1, 2, 3),
        Array[Double](1, 9, -1),
        Array[Double](0, 0, 1),
        Array[Double](0, 1, 0)
      ), 2), // row-major, laid out as is
      Seq(2, 2),
      3
    )

    val result = testMat.horzcat(testMat)
    assert(result.numRows() === 4, "horzcat() returns a result with incorrect row count")
    assert(result.numCols() === 6, "horzcat() returns a result with incorrect col count")
    assert(result.collect() === DenseMatrix((1, 2, 3, 1, 2, 3), (1, 9, -1, 1, 9, -1), (0, 0, 1, 0, 0, 1), (0, 1, 0, 0, 1, 0)),
     "horzcat() returns incorrect answer")
  }

  test("collect") {
    sc = new SparkContext("local", "test")
    val matrixParts = (0 until 200).map { i =>
      DenseMatrix.rand(50, 10)
    }
    val r = RowPartitionedMatrix.fromMatrix(sc.parallelize(matrixParts, 200))
    val rL = matrixParts.reduceLeftOption((a, b) => DenseMatrix.vertcat(a, b)).getOrElse(new DenseMatrix[Double](0, 0))
    val rD = r.collect()
    assert(rL == rD)
  }


}
