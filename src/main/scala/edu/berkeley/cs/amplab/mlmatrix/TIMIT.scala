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
    val timitTrainRDD = Utils.loadMatrixFromFile(sc, timitTrainFilename, parts)
    val timitTestRDD = Utils.loadMatrixFromFile(sc, timitTestFilename, parts)
    val timitBRDD = Utils.loadMatrixFromFile(sc, timitBFilename, parts)
    //println("TIMIT train partitions size: " + timitTrainRDD.partitions.size)
    //println("TIMIT b partitions size: " + timitBRDD.partitions.size)
    // Rows per partition
    //println("Rows per Timit train partition " + timitTrainRDD.mapPartitions(iter => Iterator.single(iter.length)).collect().mkString(" "))
    //println("Rows per Timit b partition " + timitBRDD.mapPartitions(iter => Iterator.single(iter.length)).collect().mkString(" "))

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

    println("Norm of timt B "+ timitB.normFrobenius())

    val timitResidual = Utils.computeResidualNormWithL2(timitTrain, timitB, timitX, lambda)
    println("Finished computing the residuals " + timitResidual)

    println("Condition number, residual norm, time")
    val timitR = timitTrain.qrR()
    println("Timit: " + timitTrain.condEst(Some(timitR)) + " " + timitResidual + " " + timitTime)
    println("SVDs of timitTrain " + timitTrain.svds(Some(timitR)).toArray.mkString(" "))

    val testError = Utils.calcTestErr(timitTest, timitX, timitActual, 1)
    println(testError)
  }
}
