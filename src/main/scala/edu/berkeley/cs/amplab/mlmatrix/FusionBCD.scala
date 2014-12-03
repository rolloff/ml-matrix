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


object FusionBCD extends Logging with Serializable {


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

  def loadMatrixFromFile(sc: SparkContext, filename: String): RDD[Array[Double]] = {
    sc.textFile(filename).map { line =>
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


  def calcTestErr(daisyTest: RowPartitionedMatrix, lcsTest: RowPartitionedMatrix,
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

    val predictedLabels = topKClassifier(10, fusedPrediction)
    val errPercent = getErrPercent(predictedLabels, actualLabels, numTestImages)
    errPercent
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
        println("Usage: Fusion <master> <data_dir> <parts> <solver: tsqr|normal|sgd|local> <lambda> [<stepsize> <numIters> <miniBatchFraction>]")
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
    val daisyTrainFilenames = (1 to 2).map( i => directory + "daisy-aPart" + i.toString + "-" + i.toString + "/")
    val daisyTestFilenames = (1 to 2).map( i => directory + "daisy-testFeatures-test-" + i.toString + "/")
    val daisyBFilename = directory + "daisy-null-labels/"

    // LCS filenames
    val lcsTrainFilenames = (1 to 2).map(i => directory + "lcs-aPart" + i.toString + "-" + i.toString + "/")
    val lcsTestFilenames = (1 to 2).map(i => directory + "lcs-testFeatures-test-" + i.toString + "/")
    val lcsBFilename = directory + "lcs-null-labels/"

    // Actual labels from imagenet
    val imagenetTestLabelsFilename = directory + "imagenet-test-actual/"

    // load matrix RDDs
    var daisyTrainRDDs = daisyTrainFilenames.map {
      daisyTrainFilename => loadMatrixFromFile(sc, daisyTrainFilename)
    }
    var daisyTestRDDs = daisyTestFilenames.map {
      daisyTestFilename => loadMatrixFromFile(sc, daisyTestFilename)
    }
    var daisyBRDD = loadMatrixFromFile(sc, daisyBFilename)

    var lcsTrainRDDs = lcsTrainFilenames.map {
      lcsTrainFilename => loadMatrixFromFile(sc, lcsTrainFilename)
    }
    var lcsTestRDDs = lcsTestFilenames.map {
      lcsTestFilename => loadMatrixFromFile(sc, lcsTestFilename)
    }
    var lcsBRDD = loadMatrixFromFile(sc, lcsBFilename)


    val hp = new HashPartitioner(parts)
    val daisyTrainRDDsPartitioned = daisyTrainRDDs.map( daisyTrainRDD => daisyTrainRDD.zipWithUniqueId.map(x => x.swap).partitionBy(hp).cache())
    daisyTrainRDDsPartitioned.foreach(rdd => rdd.count)
    val daisyBRDDPartitioned = daisyBRDD.zipWithUniqueId.map(x => x.swap).partitionBy(hp).cache()
    daisyBRDDPartitioned.count()
    val lcsTrainRDDsPartitioned = lcsTrainRDDs.map( lcsTrainRDD => lcsTrainRDD.zipWithUniqueId.map(x => x.swap).partitionBy(hp).cache())
    lcsTrainRDDsPartitioned.foreach(rdd => rdd.count)
    val lcsBRDDPartitioned = lcsBRDD.zipWithUniqueId.map(x => x.swap).partitionBy(hp).cache()
    lcsBRDDPartitioned.count()

    // daisyTrains should be a Seq[RowPartitionedMatrix]
    val daisyTrains = daisyTrainRDDsPartitioned.map(p => RowPartitionedMatrix.fromArray(p.map(_._2)).cache())
    val daisyB = RowPartitionedMatrix.fromArray(daisyBRDDPartitioned.map(_._2)).cache()
    val lcsTrains = lcsTrainRDDsPartitioned.map(p => RowPartitionedMatrix.fromArray(p.map(_._2)).cache())
    val lcsB = RowPartitionedMatrix.fromArray(lcsBRDDPartitioned.map(_._2)).cache()

    // Load text file as array of ints
    val imagenetTestLabelsRDD = sc.textFile(imagenetTestLabelsFilename).map { line =>
      line.split(",").map(x => x.toInt)
    }

    val hpTest = new HashPartitioner(16)

    // NOTE: We need to do this as test data has different number of entries per partition
    val daisyTestRDDsPartitioned = daisyTestRDDs.map(daisyTestRDD => daisyTestRDD.zipWithUniqueId.map(x => x.swap).partitionBy(hpTest).cache())
    daisyTestRDDsPartitioned.foreach(rdd => rdd.count)
    val lcsTestRDDsPartitioned = lcsTestRDDs.map(lcsTestRDD => lcsTestRDD.zipWithUniqueId.map(x => x.swap).partitionBy(hpTest).cache())
    lcsTestRDDsPartitioned.foreach(rdd => rdd.count)
    val imagenetTestLabelsRDDPartitioned = imagenetTestLabelsRDD.zipWithUniqueId.map(x => x.swap).partitionBy(hpTest).cache()
    imagenetTestLabelsRDDPartitioned.count

    val daisyTests = daisyTestRDDsPartitioned.map(p => RowPartitionedMatrix.fromArray(p.map(_._2)).cache())
    val lcsTests = lcsTestRDDsPartitioned.map(p => RowPartitionedMatrix.fromArray(p.map(_._2)).cache())
    val imagenetTestLabel = imagenetTestLabelsRDDPartitioned.map(_._2).cache()

    // Unpersist the RDDs
    daisyTrains.map(daisyTrain => daisyTrain.rdd.unpersist())
    daisyB.rdd.unpersist()
    lcsTrains.map(lcsTrain => lcsTrain.rdd.unpersist())
    lcsB.rdd.unpersist()
    daisyTests.map(daisyTest => daisyTest.rdd.unpersist())
    lcsTests.map(lcsTest => lcsTest.rdd.unpersist())



    val rowPartitionedSolver = getSolver(solver, numIterationsSGD, stepSize, miniBatchFraction)

    // Solve for daisy x
    var begin = System.nanoTime()
    // returns Seq[Seq[DenseMatrix[Double]]]
    val daisyXs = new BlockCoordinateDescent().solveLeastSquaresWithL2(daisyTrains, daisyB, Array(lambda), numIterationsBCD, rowPartitionedSolver).map(p => p(0))
    var end = System.nanoTime()
    // Timing numbers are in ms
    val daisyTime = (end - begin) / 1e6

    println("Finished solving for daisyXs")

    // Solve for lcs x
    var begin2 = System.nanoTime()
    val lcsXs = new BlockCoordinateDescent().solveLeastSquaresWithL2(daisyTrains, daisyB, Array(lambda), numIterationsBCD, rowPartitionedSolver).map(p => p(0))
    var end2 = System.nanoTime()
    val lcsTime = (end2 -begin2) /1e6

    println("Finished solving for lcsXs")

    val daisyResiduals = daisyTrains.zip(daisyXs).map { case (daisyTrain, daisyX) =>
      computeResidualNormWithL2(daisyTrain, daisyB, daisyX, lambda)
    }
    val lcsResiduals = lcsTrains.zip(lcsXs).map { case (lcsTrain, lcsX) =>
      computeResidualNormWithL2(lcsTrain, lcsB, lcsX, lambda)
    }
    println("Daisy Residuals " + daisyResiduals.mkString(" "))
    println("LCS Residuals " + lcsResiduals.mkString(" "))


    val daisyConditionNumbers = daisyTrains.map(daisyTrain => daisyTrain.condEst())
    val lcsConditionNumbers = lcsTrains.map(lcsTrain => lcsTrain.condEst())

    println("DaisyConditionNumbers : " + daisyConditionNumbers)
    println("LCSConditionNumbers : " + lcsConditionNumbers)

    println("DaisyTime, lcsTime " + (daisyTime, lcsTime))

    val testErrors = ((daisyTests.zip(lcsTests)).zip(daisyXs.zip(lcsXs))).map { p =>
      val daisyTest = p._1._1
      val lcsTest = p._1._2
      val daisyX = p._2._1
      val lcsX = p._2._2
      calcTestErr(daisyTest, lcsTest, daisyX, lcsX, imagenetTestLabel, 0.5, 0.5)
    }
  println("Test errors : " + testErrors.mkString(" "))
  }
}
