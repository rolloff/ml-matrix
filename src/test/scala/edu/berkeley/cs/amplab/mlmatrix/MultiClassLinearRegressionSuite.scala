package edu.berkeley.cs.amplab.mlmatrix

import java.util.concurrent.ThreadLocalRandom

import org.scalatest.FunSuite
import org.apache.spark.SparkContext
import edu.berkeley.cs.amplab.mlmatrix.util.Utils

import breeze.linalg._
import breeze.numerics._

class MultiClassLinearRegressionSuite extends FunSuite with LocalSparkContext {

  test("Test SGD") {
    sc = new SparkContext("local", "test")
    val A = RowPartitionedMatrix.createRandom(sc, 128, 16, 4, cache=true)
    val b =  A.mapPartitions(
      part => DenseMatrix.rand(part.rows, 4)).cache()

    val localA = A.collect()
    val localB = b.collect()

    val inputData = A.rdd.zip(b.rdd).flatMap { part =>
      val aMat = part._1.mat
      val bMat = part._2.mat
      val aRows = aMat.data.grouped(aMat.rows).toSeq.transpose.map(x => DenseVector(x.toArray))
      val bRows = bMat.data.grouped(bMat.rows).toSeq.transpose.map(x => x.toArray)
      bRows.zip(aRows)
    }

    val sgd = new MultiClassLinearRegressionWithSGD()
    sgd.optimizer.setStepSize(0.1)
                 .setNumIterations(50)
                 .setMiniBatchFraction(1.0)
                 .setRegParam(0.0) 

    val x = sgd.run(inputData).weights

    val localX = localA \ localB

    // println("X was " + x)
    // println("localX was " + localX)

    println("Abs max " + max(abs((x - localX).toDenseVector)))
    println("2 norm diff / norm x = " + norm((x - localX).toDenseVector, 2.0) /
      norm(localX.toDenseVector, 2.0))
  }

  test("Test Least squares SGD") {
    sc = new SparkContext("local", "test")
    val A = RowPartitionedMatrix.createRandom(sc, 128, 16, 4, cache=true)
    val b =  A.mapPartitions(
      part => DenseMatrix.rand(part.rows, 4)).cache()

    val localA = A.collect()
    val localB = b.collect()

    val sgd = new LeastSquaresGradientDescent(50, stepSize=0.1, miniBatchFraction=1.0)
    val x = sgd.solveLeastSquares(A, b)
    val localX = localA \ localB

    // println("X was " + x)
    // println("localX was " + localX)

    println("Abs max " + max(abs((x - localX).toDenseVector)))
    println("2 norm diff / norm x = " + norm((x - localX).toDenseVector, 2.0) /
      norm(localX.toDenseVector, 2.0))
  }

}
