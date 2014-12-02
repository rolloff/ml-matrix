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


  // def textFileWithName(sc: SparkContext, filename: String) = {
  //   sc.hadoopFile(filename, classOf[TextInputFormat], classOf[LongWritable], 
  //       classOf[Text]).asInstanceOf[HadoopRDD[LongWritable, Text]].mapPartitionsWithInputSplit { case (split, lineIter) =>
  //     val fileSplit = split.asInstanceOf[FileSplit]
  //     val name = fileSplit.getPath().getName()
  // 
  //     lineIter.map { line =>
  //       (name, line._2.toString)
  //     }
  //   }
  // }

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

    val predictedLabels = topKClassifier(5, fusedPrediction)
    println("Got predicted labels ")
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
    val daisyTrainFilename = directory + "daisy-aPart1-1/"
    val daisyTestFilename = directory + "daisy-testFeatures-test-1/"
    val daisyBFilename = directory + "daisy-null-labels/"

    // LCS filenames
    val lcsTrainFilename = directory + "lcs-aPart1-1/"
    val lcsTestFilename = directory + "lcs-testFeatures-test-1/"
    val lcsBFilename = directory + "lcs-null-labels/"

    // Actual labels from imagenet
    val imagenetTestLabelsFilename = directory + "imagenet-test-actual/"

    // Load data as RowPartitionedMatrices
    val daisyTrainRDD = loadMatrixFromFile(sc, daisyTrainFilename)
    val daisyTestRDD = loadMatrixFromFile(sc, daisyTestFilename)
    val daisyBRDD = loadMatrixFromFile(sc, daisyBFilename)


    val lcsTrainRDD = loadMatrixFromFile(sc, lcsTrainFilename)
    var lcsTestRDD = loadMatrixFromFile(sc, lcsTestFilename)
    val lcsBRDD = loadMatrixFromFile(sc, lcsBFilename)

    // val coalescer = Utils.createCoalescer(daisyTrainRDD, parts)

    var daisyZipped = daisyTrainRDD.zip(daisyBRDD)
    var lcsZipped = lcsTrainRDD.zip(lcsBRDD)
    val trainZipped = daisyZipped.zip(lcsZipped).repartition(parts).cache()

    // Lets cache and assert a few things 
    trainZipped.count

    val daisyTrain = RowPartitionedMatrix.fromArray(trainZipped.map(p => p._1._1)).cache()
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

    // Create a new coalescer for test data.
    // NOTE: We need to do this as test data has different number of entries per partition
    // val testCoalescer = Utils.createCoalescer(daisyTestRDD, parts)
    val testZipped = daisyTestRDD.zip(lcsTestRDD).zip(imagenetTestLabelsRDD).repartition(16).cache()

    // Lets cache and assert a few things 
    testZipped.count

    val daisyTest = RowPartitionedMatrix.fromArray(testZipped.map(p => p._1._1)).cache()
    val lcsTest = RowPartitionedMatrix.fromArray(testZipped.map(p => p._1._2)).cache()
    // NOTE: Test labels is partitioned the same way as test features
    val imagenetTestLabels = testZipped.map(p => p._2).cache()

    daisyTest.rdd.count
    lcsTest.rdd.count
    imagenetTestLabels.count
    println("imageNet coalesced")

    testZipped.unpersist()

    // Solve for daisy x
    var begin = System.nanoTime()
    val daisyX = solveForX(daisyTrain, daisyB, solver, lambda, numIterations, stepSize, miniBatchFraction)
    var end = System.nanoTime()
    // Timing numbers are in ms
    val daisyTime = (end - begin) / 1e6

    println("Finished solving for daisy X ")

    // Solve for lcs x
    var begin2 = System.nanoTime()
    val lcsX = solveForX(lcsTrain, lcsB, solver, lambda, numIterations, stepSize, miniBatchFraction)
    var end2 = System.nanoTime()
    val lcsTime = (end2 -begin2) /1e6

    println("Finished solving for lcsX")

    // FIXME: Residual norm needs to be calculated for regularized problem

    // Information about the spectrum of the matrices
    // println("Condition number of daisyTrain " + daisyTrain.condEst())
    // println("Condition number of daisyTest " + daisyTest.condEst())
    // println("Condition number of lcsTrain " + lcsTrain.condEst())
    // println("Condition number of lcsTest " + lcsTest.condEst())

    val daisyResidual = computeResidualNormWithL2(daisyTrain, daisyB, daisyX, lambda)
    val lcsResidual = computeResidualNormWithL2(lcsTrain, lcsB, lcsX, lambda)
    println("Finished computing the residuals " + daisyResidual + " " + lcsResidual)

    println("Condition number, residual norm, time")
    val daisyR = daisyTrain.qrR()
    println("Daisy: " + daisyTrain.condEst(Some(daisyR)) + " " + daisyResidual + " " + daisyTime)
    println("SVDs of daisyTrain " + daisyTrain.svds(Some(daisyR)).toArray.mkString(" "))

    val lcsR = lcsTrain.qrR()
    println("LCS: " + lcsTrain.condEst(Some(lcsR)) + " " + lcsResidual + " " + lcsTime)
    println("SVDs of lcsTrain " + lcsTrain.svds(Some(lcsR)).toArray.mkString(" "))

    val testError = calcTestErr(daisyTest, lcsTest, daisyX, lcsX, imagenetTestLabels, 0.5, 0.5)
    println(testError)
  }
}
