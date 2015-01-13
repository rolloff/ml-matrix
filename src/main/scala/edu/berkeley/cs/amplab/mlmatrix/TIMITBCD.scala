package edu.berkeley.cs.amplab.mlmatrix

import scala.io.Source._

import breeze.linalg._
import breeze.numerics._

import org.apache.hadoop.mapred.FileSplit
import org.apache.hadoop.mapred.{FileInputFormat, InputFormat, JobConf, SequenceFileInputFormat, TextInputFormat}
import org.apache.hadoop.io.{ArrayWritable, BooleanWritable, BytesWritable, DoubleWritable, FloatWritable, IntWritable, LongWritable, NullWritable, Text, Writable}

import edu.berkeley.cs.amplab.mlmatrix.util.QRUtils
import edu.berkeley.cs.amplab.mlmatrix.util.Utils

import org.apache.spark.HashPartitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.HadoopRDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object TimitBCD extends Logging with Serializable {


  def solveForX(
      A: RowPartitionedMatrix,
      b: RowPartitionedMatrix,
      solver: String,
      lambda: Double,
      numIterationsSGD: Integer,
      stepSize: Double,
      miniBatchFraction: Double) = {
    solver.toLowerCase match {
      case "normal" =>
        new NormalEquations().solveLeastSquaresWithL2(A, b, lambda)
      case "sgd" =>
        new LeastSquaresGradientDescent(numIterationsSGD, stepSize, miniBatchFraction).solveLeastSquaresWithL2(A, b, lambda)
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

  def getSolver(solver: String, numIterationsSGD: Int , stepSize: Double, miniBatchFraction: Double) = {
    solver.toLowerCase match {
      case "normal" =>
        new NormalEquations()
      case "sgd" =>
        new LeastSquaresGradientDescent(numIterationsSGD.get, stepSize.get, miniBatchFraction.get)
      case "tsqr" =>
        new TSQR()
      case _ =>
        logError("Invalid Solver " + solver + " should be one of tsqr|normal|sgd")
        logError("Using TSQR")
        new TSQR()
    }
  }

  def main(args: Array[String]) {
    if (args.length < 5) {
      println("Got args " + args.mkString(" "))
      println("Usage: Fusion <master> <data_dir> <parts> <solver: tsqr|normal|sgd|local> <lambda> <numIterationsBCD> [<stepsize> <numIters> <miniBatchFraction>]")
      System.exit(0)
    }
    val sparkMaster = args(0)
    //Directory that holds the data
    val directory = args(1)
    val parts = args(2).toInt
    val solver = args(3)
    // Lambda for regularization
    val lambda = args(4).toDouble
    val numIterationsBCD = args(5).toInt

    var stepSize = 0.1
    var numIterationsSGD = 10
    var miniBatchFraction = 1.0
    if (solver == "sgd") {
      if (args.length < 9) {
        println("Usage: TIMIT <master> <data_dir> <parts> <solver: tsqr|normal|sgd|local> <lambda> [<stepsize> <numIters> <miniBatchFraction>]")
        System.exit(0)
      } else {
        stepSize = args(6).toDouble
        numIterationsSGD = args(7).toInt
        miniBatchFraction = args(8).toDouble
      }
    }

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("Fusion")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)

    // Daisy filenames
    val trainFilenames = (1 to 5).map( i => directory + "timit-fft-aPart" + i.toString + "-" + i.toString + "/")
    val testFilenames = (1 to 5).map( i => directory + "timit-fft-testRPM-test-" + i.toString + "/")
    val bFilename = directory + "timit-fft-null-labels/"

    // Actual labels from imagenet
    val actualLabelsFilename = directory + "timit-actual/"

    // load matrix RDDs
    val trainRDDs = trainFilenames.map {
      trainFilename => Utils.loadMatrixFromFile(sc, trainFilename, parts)
    }
    val testRDDs = testFilenames.map {
      testFilename => Utils.loadMatrixFromFile(sc, testFilename, parts)
    }
    val bRDD = Utils.loadMatrixFromFile(sc, bFilename, parts)



    val hp = new HashPartitioner(parts)
    val trainRDDsPartitioned = trainRDDs.map { trainRDD =>
      Utils.repartitionAndSortWithinPartitions(trainRDD.zipWithUniqueId.map(x => x.swap), hp).cache()
    }
    trainRDDsPartitioned.foreach(rdd => rdd.count)
    // trains should be a Seq[RowPartitionedMatrix]
    val trains = trainRDDsPartitioned.map(p => RowPartitionedMatrix.fromArray(p.map(_._2)).cache())
    trains.map(train => train.rdd.count)
    trainRDDsPartitioned.map(train => train.unpersist())

    val bRDDPartitioned = Utils.repartitionAndSortWithinPartitions(
      bRDD.zipWithUniqueId.map(x => x.swap), hp).cache()
    bRDDPartitioned.count()
    val b = RowPartitionedMatrix.fromArray(bRDDPartitioned.map(_._2)).cache()
    b.rdd.count
    bRDDPartitioned.unpersist()

    // Load text file as array of ints
    val actualLabelsRDD = sc.textFile(actualLabelsFilename).map { line =>
      line.split(",").map(x => x.toInt)
    }

    val hpTest = new HashPartitioner(16)

    // NOTE: We need to do this as test data has different number of entries per partition
    val testRDDsPartitioned = testRDDs.map { testRDD =>
      Utils.repartitionAndSortWithinPartitions(testRDD.zipWithUniqueId.map(x => x.swap), hpTest).cache()
    }
    testRDDsPartitioned.foreach(rdd => rdd.count)
    val tests = testRDDsPartitioned.map(p => RowPartitionedMatrix.fromArray(p.map(_._2)).cache())
    tests.map(test => test.rdd.count())
    testRDDsPartitioned.map(train => train.unpersist())

    val actualLabelsRDDPartitioned = Utils.repartitionAndSortWithinPartitions(
      actualLabelsRDD.zipWithUniqueId.map(x => x.swap), hpTest).cache()
    actualLabelsRDDPartitioned.count

    val actualLabels = actualLabelsRDDPartitioned.map(_._2).cache()
    actualLabels.count
    actualLabelsRDDPartitioned.unpersist()

    val rowPartitionedSolver = getSolver(solver, numIterationsSGD, stepSize, miniBatchFraction)

    // Solve for daisy x
    var begin = System.nanoTime()
    // returns Seq[Seq[DenseMatrix[Double]]]
    val xs = new BlockCoordinateDescent().solveLeastSquaresWithL2(trains, b, Array(lambda), numIterationsBCD, rowPartitionedSolver).map(p => p(0))
    var end = System.nanoTime()
    // Timing numbers are in ms
    val time = (end - begin) / 1e6

    println("Finished solving for xs")

    val testErrors = Utils.calcTestErrors(tests, xs, actualLabels, 1)
    println("Test errors : " + testErrors)

    val residuals = Utils.computeResidualNormsWithL2(trains, b, xs, lambda)

    println("Residuals " + residuals.mkString(" "))
    //val conditionNumbers = trains.map(train => train.condEst())
    //println("ConditionNumbers : " + conditionNumbers)
    println("Time: " + time)
  }
}
