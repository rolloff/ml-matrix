package edu.berkeley.cs.amplab.mlmatrix

import scala.io.Source._

import breeze.linalg._
import breeze.numerics._

import org.apache.hadoop.mapred.FileSplit
import org.apache.hadoop.mapred.{FileInputFormat, InputFormat, JobConf, SequenceFileInputFormat, TextInputFormat}
import org.apache.hadoop.io.{ArrayWritable, BooleanWritable, BytesWritable, DoubleWritable, FloatWritable, IntWritable, LongWritable, NullWritable, Text, Writable}

import edu.berkeley.cs.amplab.mlmatrix.util.QRUtils
import edu.berkeley.cs.amplab.mlmatrix.util.Utils

import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.HadoopRDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object TIMIT extends Logging with Serializable {


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

  def computeResidualNorm(A: RowPartitionedMatrix,
      b: RowPartitionedMatrix,
      xComputed: DenseMatrix[Double]) = {
    val xBroadcast = A.rdd.context.broadcast(xComputed)
    val axComputed = A.mapPartitions { part =>
      part*xBroadcast.value
    }
    val residualNorm = (b - axComputed).normFrobenius()
    residualNorm
  }

  def computeResidualNormWithL2(A: RowPartitionedMatrix,
      b: RowPartitionedMatrix,
      xComputed: DenseMatrix[Double], lambda: Double) = {
    val unregularizedNorm = computeResidualNorm(A,b,xComputed)
    val normX = norm(xComputed.toDenseVector)

    scala.math.sqrt(unregularizedNorm*unregularizedNorm + lambda*normX*normX)
  }


  def loadMatrixFromFile(sc: SparkContext, filename: String, parts: Int): RDD[Array[Double]] = {
    sc.textFile(filename, parts).map { line =>
      line.split(",").map(y => y.toDouble)
    }
  }

  def getErrPercent(predicted: RDD[Array[Int]], actual: RDD[Array[Int]], numTestImages: Int): Double = {
    // FIXME: Each image only has one actual label, so actual should be an RDD[Int]
    val totalErr = predicted.zip(actual).map({ case (topKLabels, actualLabel) =>
      if (topKLabels.contains(actualLabel(0))) {
        0.0
      } else {
        1.0
      }
    }).reduce(_ + _)

    val errPercent = totalErr / numTestImages * 100.0
    errPercent
  }

  def topKClassifier(k: Int, in: RDD[Array[Double]]) : RDD[Array[Int]] = {
    // Returns top k indices with maximum value
    in.map { ary =>
      ary.toSeq.zipWithIndex.sortBy(_._1).takeRight(k).map(_._2).toArray
    }
    // in.map { ary => argtopk(ary, k).toArray }
  }


  def calcTestErr(test: RowPartitionedMatrix,
    x: DenseMatrix[Double],
    actualLabels: RDD[Array[Int]]): Double = {

    // Compute number of test images
    val numTestImages = test.numRows().toInt

    // Broadcast x
    val xBroadcast = test.rdd.context.broadcast(x)

    // Calculate predictions
    val prediction = test.rdd.map { mat =>
      mat.mat * xBroadcast.value
    }

    val predictionArray = prediction.flatMap { p =>
      p.data.grouped(p.rows).toSeq.transpose.map(y => y.toArray)
    }

    val predictedLabels = topKClassifier(5, predictionArray)
    val errPercent = getErrPercent(predictedLabels, actualLabels, numTestImages)
    errPercent
  }

  def main(args: Array[String]) {
    if (args.length < 5) {
      println("Got args " + args.mkString(" "))
      println("Usage: Fusion <master> <data_dir> <parts> <solver: tsqr|normal|sgd|local> <lambda> [<stepsize> <numIters> <miniBatchFraction>]")
      System.exit(0)
    }
    val sparkMaster = args(0)
    //Directory that holds the data
    val directory = args(1)
    val parts = args(2).toInt
    val solver = args(3)
    // Lambda for regularization
    val lambda = args(4).toDouble

    var stepSize = 0.1
    var numIterations = 10
    var miniBatchFraction = 1.0
    if (solver == "sgd") {
      if (args.length < 8) {
        println("Usage: Fusion <master> <data_dir> <parts> <solver: tsqr|normal|sgd|local> <lambda> [<stepsize> <numIters> <miniBatchFraction>]")
        System.exit(0)
      } else {
        stepSize = args(5).toDouble
        numIterations = args(6).toInt
        miniBatchFraction = args(7).toDouble
      }
    }

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("Fusion")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)

    // Daisy filenames
    val timitTrainFilename = directory + "timit-fft-aPart1-1/"
    val timitTestFilename = directory + "timit-fft-testRPM-test-1/"
    val timitBFilename = directory + "timit-fft-null-labels/"

    // Actual labels from imagenet
    val timitActualFilename = directory + "timit-actual/"

    // load matrix RDDs
    val timitTrainRDD = loadMatrixFromFile(sc, timitTrainFilename, parts)
    val timitTestRDD = loadMatrixFromFile(sc, timitTestFilename, parts)
    val timitBRDD = loadMatrixFromFile(sc, timitBFilename, parts)
    println("TIMIT train partitions size: " + timitTrainRDD.partitions.size)
    println("TIMIT b partitions size: " + timitBRDD.partitions.size)
    // Rows per partition
    println("Rows per Timit train partition " + timitTrainRDD.mapPartitions(iter => Iterator.single(iter.length)).collect().mkString(" "))
    println("Rows per Timit b partition " + timitBRDD.mapPartitions(iter => Iterator.single(iter.length)).collect().mkString(" "))

    var timitZipped = timitTrainRDD.zip(timitBRDD).repartition(parts).cache()



    // Lets cache and assert a few things
    timitZipped.count

    val timitTrain = RowPartitionedMatrix.fromArray(timitZipped.map(p => p._1)).cache()
    val timitB = RowPartitionedMatrix.fromArray(timitZipped.map(p => p._2)).cache()

    timitTrain.rdd.count
    timitB.rdd.count


    // Load text file as array of ints
    val timitActualRDD = sc.textFile(timitActualFilename).map { line =>
      line.split(",").map(x => x.toInt)
    }

    // NOTE: We need to do this as test data has different number of entries per partition
    val testZipped = timitTestRDD.zip(timitActualRDD).repartition(16).cache()

    // Lets cache and assert a few things
    testZipped.count

    val timitTest = RowPartitionedMatrix.fromArray(testZipped.map(p => p._1)).cache()
    // NOTE: Test labels is partitioned the same way as test features
    val timitActual = testZipped.map(p => p._2).cache()

    timitTest.rdd.count
    timitActual.count

    // FIXME: Unpersist RDDs right after we cache them?

    // Solve for timit x
    var begin = System.nanoTime()
    val timitX = solveForX(timitTrain, timitB, solver, lambda, numIterations, stepSize, miniBatchFraction)
    var end = System.nanoTime()
    // Timing numbers are in ms
    val timitTime = (end - begin) / 1e6

    println("Finished solving for timit X ")


    // Information about the spectrum of the matrices
    // println("Condition number of timitTrain " + timitTrain.condEst())
    // println("Condition number of timitTest " + timitTest.condEst())

    val timitResidual = computeResidualNormWithL2(timitTrain, timitB, timitX, lambda)
    println("Finished computing the residuals " + timitResidual)

    println("Condition number, residual norm, time")
    val timitR = timitTrain.qrR()
    println("Timit: " + timitTrain.condEst(Some(timitR)) + " " + timitResidual + " " + timitTime)
    println("SVDs of timitTrain " + timitTrain.svds(Some(timitR)).toArray.mkString(" "))

    val testError = calcTestErr(timitTest, timitX, timitActual)
    println(testError)
  }
}
