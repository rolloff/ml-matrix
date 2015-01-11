package edu.berkeley.cs.amplab.mlmatrix

import scala.io.Source._

import breeze.linalg._
import breeze.numerics._

import org.apache.hadoop.mapred.FileSplit
import org.apache.hadoop.mapred.TextInputFormat
import org.apache.hadoop.io.{LongWritable, Text, Writable}

import edu.berkeley.cs.amplab.mlmatrix.util.QRUtils
import edu.berkeley.cs.amplab.mlmatrix.util.Utils

import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.HadoopRDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object Fusion extends Logging with Serializable {

  def solveForX(
      A: RowPartitionedMatrix,
      b: RowPartitionedMatrix,
      solver: String,
      lambda: Double,
      numIterations: Integer,
      stepSize: Double,
      miniBatchFraction: Double) = {
    solver.toLowerCase match {
      case "normal" =>
        new NormalEquations().solveLeastSquaresWithL2(A, b, lambda)
      case "sgd" =>
        new LeastSquaresGradientDescent(numIterations, stepSize, miniBatchFraction).solveLeastSquaresWithL2(A, b, lambda)
      case "tsqr" =>
        new TSQR().solveLeastSquaresWithL2(A, b, lambda)
      case "local" =>
        // Solve regularized least squares problem with local qr factorization
        val (aTilde, bTilde) = QRUtils.qrSolveWithL2(A.collect(), b.collect(), lambda)
        aTilde \ bTilde
      case _ =>
        logError("Invalid Solver " + solver + " should be one of tsqr|normal|sgd")
        logError("Using TSQR")
        new TSQR().solveLeastSquares(A, b)
    }
  }

  def calcFusedTestErr(daisyTest: RowPartitionedMatrix, lcsTest: RowPartitionedMatrix,
    daisyX: DenseMatrix[Double], lcsX: DenseMatrix[Double],
    actualLabels: RDD[Array[Int]],
    daisyWt: Double, lcsWt: Double): Double = {

    // Compute number of test images
    val numTestImages = daisyTest.numRows().toInt

    // Broadcast x
    val daisyXBroadcast = daisyTest.rdd.context.broadcast(daisyX)
    val lcsXBroadcast = lcsTest.rdd.context.broadcast(lcsX)

    // Calculate predictions
    val daisyPrediction = daisyTest.rdd.map { mat =>
      mat.mat * daisyXBroadcast.value
    }
    val lcsPrediction = lcsTest.rdd.map { mat =>
      mat.mat * lcsXBroadcast.value
    }

    // Fuse b matrices
    val fusedPrediction = daisyPrediction.zip(lcsPrediction).flatMap { p =>
      val fused = (p._1*daisyWt + p._2*lcsWt)
      // Convert from DenseMatrix to Array[Array[Double]]
      fused.data.grouped(fused.rows).toSeq.transpose.map(x => x.toArray)
    }

    val predictedLabels = Utils.topKClassifier(5, fusedPrediction)
    val errPercent = Utils.getErrPercent(predictedLabels, actualLabels, numTestImages)
    errPercent
  }

  def main(args: Array[String]) {
    if (args.length < 5) {
      println("Usage: Fusion <master> <data_dir> <parts> <solver: tsqr|normal|sgd|local> <lambda> [<stepsize> <numIters> <miniBatchFraction>]")
      System.exit(0)
    }

    val sparkMaster = args(0)
    // Directory that holds the data
    val directory = args(1)
    val parts = args(2).toInt
    val solver = args(3)
    // Lambda for regularization
    val lambda = args(4).toDouble

    println("Running Fusion with ")
    println("master: " + sparkMaster)
    println("directory: " + directory)
    println("parts: " + parts)
    println("solver: " + solver)
    println("lambda: " + lambda)

    var stepSize = 0.1
    var numIterations = 10
    var miniBatchFraction = 1.0
    if (solver == "sgd") {
      if (args.length < 8) {
        println(args.mkString(","))
        println("Usage: Fusion <master> <data_dir> <parts> <solver: tsqr|normal|sgd|local> <lambda> [<stepsize> <numIters> <miniBatchFraction>]")
        System.exit(0)
      } else {
        stepSize = args(5).toDouble
        numIterations = args(6).toInt
        miniBatchFraction = args(7).toDouble

        println("stepSize: " + stepSize)
        println("numIterations: " + numIterations)
        println("miniBatchFraction: " + miniBatchFraction)
      }
    }

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("Fusion")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)
    sc.addSparkListener(new org.apache.spark.scheduler.JobLogger())

    // Daisy filenames
    val daisyTrainFilename = directory + "daisy-aPart1-1/"
    val daisyTestFilename = directory + "daisy-testFeatures-test-1/"
    val daisyBFilename = directory + "daisy-null-labels/"

    // LCS filenames
    val lcsTrainFilename = directory + "lcs-aPart1-1/"
    val lcsTestFilename = directory + "lcs-testFeatures-test-1/"
    val lcsBFilename = directory + "lcs-null-labels/"

    // Actual labels from imagenet
    val imagenetTestLabelsFilename = directory + "imagenet-test-actual/"

    // load matrix RDDs
    val daisyTrainRDD = Utils.loadMatrixFromFile(sc, daisyTrainFilename, parts)
    val daisyTestRDD = Utils.loadMatrixFromFile(sc, daisyTestFilename, parts)
    val daisyBRDD = Utils.loadMatrixFromFile(sc, daisyBFilename, parts)

    val lcsTrainRDD = Utils.loadMatrixFromFile(sc, lcsTrainFilename, parts)
    var lcsTestRDD = Utils.loadMatrixFromFile(sc, lcsTestFilename, parts)
    val lcsBRDD = Utils.loadMatrixFromFile(sc, lcsBFilename, parts)


    var daisyZipped = daisyTrainRDD.zip(daisyBRDD)
    var lcsZipped = lcsTrainRDD.zip(lcsBRDD)
    val trainZipped = daisyZipped.zip(lcsZipped).repartition(parts).cache()

    // Lets cache and assert a few things
    trainZipped.count

    var daisyTrain = RowPartitionedMatrix.fromArray(trainZipped.map(p => p._1._1)).cache()
    val daisyB = RowPartitionedMatrix.fromArray(trainZipped.map(p => p._1._2)).cache()
    val lcsTrain = RowPartitionedMatrix.fromArray(trainZipped.map(p => p._2._1)).cache()
    val lcsB = RowPartitionedMatrix.fromArray(trainZipped.map(p => p._2._2)).cache()

    daisyTrain.rdd.count
    daisyB.rdd.count
    lcsTrain.rdd.count
    lcsB.rdd.count

    trainZipped.unpersist()

    // Load text file as array of ints
    val imagenetTestLabelsRDD = sc.textFile(imagenetTestLabelsFilename).map { line =>
      line.split(",").map(x => x.toInt)
    }

    // NOTE: We need to do this as test data has different number of entries per partition
    val testZipped = daisyTestRDD.zip(lcsTestRDD).zip(imagenetTestLabelsRDD).repartition(16).cache()

    // Lets cache and assert a few things
    testZipped.count

    var daisyTest = RowPartitionedMatrix.fromArray(testZipped.map(p => p._1._1)).cache()
    val lcsTest = RowPartitionedMatrix.fromArray(testZipped.map(p => p._1._2)).cache()
    // NOTE: Test labels is partitioned the same way as test features

    val imagenetTestLabels = testZipped.map(p => p._2).cache()

    daisyTest.rdd.count
    lcsTest.rdd.count
    imagenetTestLabels.count
    println("imageNet coalesced")

    testZipped.unpersist()

    // Remove unecessary columns from Daisy
    //val colIndices = fromFile(directory + "daisy-0-column-indices.txt").getLines.map(x=>x.toInt).toList
    //daisyTrain = daisyTrain.delete(colIndices, Axis._1)
    //daisyTest = daisyTest.delete(colIndices, Axis._1)

    // Compute daisy and LCS row norms
    /*
    val daisyRowNorms = daisyTrain.rdd.flatMap { part =>
      val nRows = part.mat.rows
      (0 until nRows).map { row =>
        norm(part.mat(row, ::).inner, 2)
      }
    }.collect().sorted

    println("DAISY Row norms highest " + daisyRowNorms.takeRight(10).mkString(","))

    val lcsRowNorms = lcsTrain.rdd.flatMap { part =>
      val nRows = part.mat.rows
      (0 until nRows).map { row =>
        norm(part.mat(row, ::).inner, 2)
      }
    }.collect().sorted

    println("LCS Row norms highest " + lcsRowNorms.takeRight(10).mkString(","))
    */

    println("daisyTrain rows " + daisyTrain.numRows() + ", daisyTrain cols " + daisyTrain.numCols())
    println("lcsTrain rows " + lcsTrain.numRows() + ", lcsTrain cols " + lcsTrain.numCols())

    // Combine problems
    val train = daisyTrain.horzcat(lcsTrain)
    val test = daisyTest.horzcat(lcsTest)
    println("train rows " + train.numRows() + ", train cols " + train.numCols())

    // FIXME: figure out what you are printing
    /*
    val rowsPerPartition = train.rdd.mapPartitionsWithIndex { case (part, iter) =>
      if (iter.isEmpty) {
        Iterator()
      } else {
        iter.zipWithIndex.map(x => (part, x._2, x._1.mat.rows.toLong))
      }
    }.collect().sortBy(x => (x._1, x._2))
    */
    val rowsPerPartition = train.rdd.map { part =>
      part.mat.rows.toLong
    }.collect()
    println(rowsPerPartition.slice(0, 10).mkString(" "))


    // Solve for x
    var begin = System.nanoTime()
    val x = solveForX(train, daisyB, solver, lambda, numIterations, stepSize, miniBatchFraction)
    var end = System.nanoTime()
    val time = (end - begin) / 1e6
    println("Finished solving for x in " + time + " ms" )
    val residual = Utils.computeResidualNormWithL2(train, daisyB, x, lambda)
    println("Combined solve residual is " + residual)
    val combinedTestError = Utils.calcTestErr(test, x, imagenetTestLabels)
    println("Combined solve test error is " + combinedTestError)


    // Solve for daisy x
    begin = System.nanoTime()
    val daisyX = solveForX(daisyTrain, daisyB, solver, lambda, numIterations, stepSize, miniBatchFraction)
    end = System.nanoTime()
    // Timing numbers are in ms
    val daisyTime = (end - begin) / 1e6
    println("Finished solving for daisy X in" + daisyTime + " ms")

    // Solve for lcs x
    begin = System.nanoTime()
    val lcsX = solveForX(lcsTrain, lcsB, solver, lambda, numIterations, stepSize, miniBatchFraction)
    end = System.nanoTime()
    val lcsTime = (end -begin) /1e6
    println("Finished solving for lcsX in" + lcsTime + " ms")

    val daisyResidual = Utils.computeResidualNormWithL2(daisyTrain, daisyB, daisyX, lambda)
    val lcsResidual = Utils.computeResidualNormWithL2(lcsTrain, lcsB, lcsX, lambda)
    println("Finished computing the residuals " + daisyResidual + " " + lcsResidual)
    val testError = calcFusedTestErr(daisyTest, lcsTest, daisyX, lcsX, imagenetTestLabels, 0.5, 0.5)
    println("Uncombined solve test error is " + testError)

  }
}
