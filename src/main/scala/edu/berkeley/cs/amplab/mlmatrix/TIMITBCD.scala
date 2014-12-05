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


  def calcTestErr(tests: Seq[RowPartitionedMatrix],
    xs: Seq[DenseMatrix[Double]],
    actualLabels: RDD[Array[Int]]): DenseVector[Double] = {

    // Compute number of test images
    val l = tests.length
    var i = 0
    val numTestImages = tests(0).numRows().toInt

    var runningSum : Option[RDD[Array[Double]]] = None
    val testErrors = DenseVector.zeros[Double](l)

    while (i < l) {
      val A = tests(i)
      val x = A.rdd.context.broadcast(xs(i))
      val Ax = A.rdd.map( mat => mat.mat*x.value)

      val prediction = Ax.flatMap { p =>
        p.data.grouped(p.rows).toSeq.transpose.map(x => x.toArray)
      }

      if (runningSum.isEmpty) {
        runningSum = Some(prediction)
      } else {
        runningSum = Some(runningSum.get.zip(prediction).map(p => p._1.zip(p._2).map( y =>
          y._1 + y._2)))
      }
      val predictedLabels = topKClassifier(5, runningSum.get)
      val errPercent = getErrPercent(predictedLabels, actualLabels, numTestImages)
      testErrors(i) = errPercent
    }
    testErrors
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
    val trainFilenames = (1 to 2).map( i => directory + "daisy-aPart" + i.toString + "-" + i.toString + "/")
    val testFilenames = (1 to 2).map( i => directory + "daisy-testFeatures-test-" + i.toString + "/")
    val bFilename = directory + "daisy-null-labels/"

    // Actual labels from imagenet
    val actualLabelsFilename = directory + "imagenet-test-actual/"

    // load matrix RDDs
    val trainRDDs = trainFilenames.map {
      trainFilename => loadMatrixFromFile(sc, trainFilename)
    }
    val testRDDs = testFilenames.map {
      testFilename => loadMatrixFromFile(sc, testFilename)
    }
    val bRDD = loadMatrixFromFile(sc, bFilename)



    val hp = new HashPartitioner(parts)
    val trainRDDsPartitioned = trainRDDs.map( trainRDD => trainRDD.zipWithUniqueId.map(x => x.swap).partitionBy(hp).cache())
    trainRDDsPartitioned.foreach(rdd => rdd.count)
    val bRDDPartitioned = bRDD.zipWithUniqueId.map(x => x.swap).partitionBy(hp).cache()
    bRDDPartitioned.count()


    // trains should be a Seq[RowPartitionedMatrix]
    val trains = trainRDDsPartitioned.map(p => RowPartitionedMatrix.fromArray(p.map(_._2)).cache())
    val b = RowPartitionedMatrix.fromArray(bRDDPartitioned.map(_._2)).cache()

    // Load text file as array of ints
    val actualLabelsRDD = sc.textFile(actualLabelsFilename).map { line =>
      line.split(",").map(x => x.toInt)
    }

    val hpTest = new HashPartitioner(16)

    // NOTE: We need to do this as test data has different number of entries per partition
    val testRDDsPartitioned = testRDDs.map(testRDD => testRDD.zipWithUniqueId.map(x => x.swap).partitionBy(hpTest).cache())
    testRDDsPartitioned.foreach(rdd => rdd.count)
    val actualLabelsRDDPartitioned = actualLabelsRDD.zipWithUniqueId.map(x => x.swap).partitionBy(hpTest).cache()
    actualLabelsRDDPartitioned.count

    val tests = testRDDsPartitioned.map(p => RowPartitionedMatrix.fromArray(p.map(_._2)).cache())

    val actualLabels = actualLabelsRDDPartitioned.map(_._2).cache()

    // Unpersist the RDDs
    trains.map(train => train.rdd.unpersist())
    b.rdd.unpersist()

    tests.map(test => test.rdd.unpersist())
    val rowPartitionedSolver = getSolver(solver, numIterationsSGD, stepSize, miniBatchFraction)

    // Solve for daisy x
    var begin = System.nanoTime()
    // returns Seq[Seq[DenseMatrix[Double]]]
    val xs = new BlockCoordinateDescent().solveLeastSquaresWithL2(trains, b, Array(lambda), numIterationsBCD, rowPartitionedSolver).map(p => p(0))
    var end = System.nanoTime()
    // Timing numbers are in ms
    val time = (end - begin) / 1e6

    println("Finished solving for xs")

    val residuals = trains.zip(xs).map { case (train, x) =>
      computeResidualNormWithL2(train, b, x, lambda)
    }

    println("Residuals " + residuals.mkString(" "))
    val conditionNumbers = trains.map(train => train.condEst())
    println("ConditionNumbers : " + conditionNumbers)
    println("Time: " + time)
    val testErrors = calcTestErr(tests, xs, actualLabels)
    println("Test errors : " + testErrors)
  }
}
