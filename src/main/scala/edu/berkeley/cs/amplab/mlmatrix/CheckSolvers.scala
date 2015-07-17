package edu.berkeley.cs.amplab.mlmatrix

import java.io.File
import java.util.concurrent.ThreadLocalRandom
import java.util.Arrays

import scala.io.Source._

import breeze.linalg._
import breeze.numerics._

import org.apache.hadoop.mapred.FileSplit
import org.apache.hadoop.mapred.TextInputFormat
import org.apache.hadoop.io.{LongWritable, Text, Writable}

import edu.berkeley.cs.amplab.mlmatrix.util.QRUtils
import edu.berkeley.cs.amplab.mlmatrix.util.Utils
import edu.berkeley.cs.amplab.mlmatrix.util.MatrixUtils

import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.HadoopRDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._


object CheckSolvers extends Logging with Serializable {

  def main(args: Array[String]) {
    if (args.length < 6) {
      println("Usage: CheckSolvers <master> <data_dir> <dataset> <parts> <solver: tsqr|normal> <lambda>")
      System.exit(0)
    }
    val sparkMaster = args(0)
    val directory = args(1)
    val dataset = args(2).toString
    val parts = args(3).toInt
    val solver = args(4).toString
    val lambda = args(5).toDouble

    println("Running CheckSolvers with ")
    println("master: " + sparkMaster)
    println("directory: " + directory)
    println("dataset: " + dataset)
    println("parts: " + parts)
    println("solver: " + solver)
    println("lambda: "  + lambda)

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("CheckSolvers")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)

    var trainFilename = directory
    var testFilename = directory
    dataset.toLowerCase match {
      case "imagenet-fv-4k" =>
        //trainFilename += "imagenet-fv-4k-trainFeatures"
        //bFilename += "imagenet-fv-4k-trainLabels"
        trainFilename += "imagenet-fv-4k-trainAll.txt"
        testFilename += "imagenet-fv-4k-testAll.txt"
      case _ =>
        logError("Invalid dataset")
        logError("Using imagenet-fv-4k")
        trainFilename += "imagenet-fv-4k-trainAll.txt"
        testFilename += "imagenet-fv-4k-testAll.txt"
    }

    val trainFileRDD= sc.textFile(trainFilename, parts).map(line=>line.split(",")).cache()
    val trainRDD = trainFileRDD.map(part=> part(0).split(" ").map(a=>a.toDouble))
    var train = RowPartitionedMatrix.fromArray(trainRDD).cache()

    val R = train.qrR()
    val lambdas = Seq(0.000025, 0.00005, 0.000075, 0.0001, 0.001, 0.01, 0.1)
    lambdas.map { lambda =>
      val gamma = DenseMatrix.eye[Double](R.rows) :* math.sqrt(lambda)
      val Rgamma = DenseMatrix.vertcat(R, gamma)
      val svd.SVD(u,s,v) = svd(Rgamma)
      println("Lambda: " + lambda + "Cond: " + s(0)/s(R.rows-1) )
    }


    /*

    val trainClasses = trainFileRDD.map(part => part(1).toInt)
    // Create matrix of +1/-1s given class labels
    // Assume classId is integer in [1,1000]
    val trainBRDD = trainClasses.map { classId =>
      val classes = Array.fill[Double](1000)(-1)
      classes(classId-1) = 1
      classes
    }
    trainFileRDD.unpersist()

    val testFileRDD = sc.textFile(testFilename, parts).map(line=>line.split(",")).cache()
    val testRDD = testFileRDD.map(part=> part(0).split(" ").map(a=>a.toDouble))
    val testLabelsRDD = testFileRDD.map(part=>Array(part(1).toInt - 1))
    testFileRDD.unpersist()


    val trainZipped = trainRDD.zip(trainBRDD).cache()
    val testZipped = testRDD.zip(testLabelsRDD).cache()
    trainZipped.count
    testZipped.count
    var train = RowPartitionedMatrix.fromArray(trainZipped.map(p=>p._1)).cache()
    var trainB = RowPartitionedMatrix.fromArray(trainZipped.map(p=>p._2)).cache()
    var test = RowPartitionedMatrix.fromArray(testZipped.map(p=>p._1)).cache()
    val testLabels = testZipped.map(p=>p._2)
    train.rdd.count
    trainB.rdd.count
    trainZipped.unpersist()
    test.rdd.count
    testLabels.count
    testZipped.unpersist()

    // Compute means
    val trainMeans = (train.reduceColElements(_+_).collect()).toDenseVector :* (1/ train.numRows().toDouble)
    val trainBMeans = (trainB.reduceColElements(_+_).collect()).toDenseVector :* (1/ trainB.numRows().toDouble)
    // Subtract trainMeans from each row of matrix
    train = train.mapPartitions(part=> part(*,::)-trainMeans)
    trainB = trainB.mapPartitions(part=> part(*, ::)-trainBMeans)



    val x= solver.toLowerCase match {
      case "normal" =>
        new NormalEquations().solveLeastSquaresWithL2(train, trainB, lambda)
      case "tsqr" =>
        new TSQR().solveLeastSquaresWithL2(train, trainB, lambda)
      case _ =>
        logError("Invalid Solver ")
        logError("Using Normal Equations")
        new NormalEquations().solveLeastSquaresWithL2(train, trainB, lambda)
    }

    // Record normA, normB, normX, norm(AX-B), norm(AX-B) + lambda*norm(X)
    val residual = Utils.computeResidualNorm(train, trainB, x)
    //val normA = train.normFrobenius()
    //val normB = trainB.normFrobenius()
    val normX = norm(x.toDenseVector)
    val residualWithRegularization = residual*residual+lambda*normX*normX
    //println("Rows of train:  " + train.numRows())
    //println("Columns of train " + train.numCols())
    println("Residual: " + residual)
    //println("normA: " + normA)
    //println("normB: " + normB)
    println("normX: " + normX)
    println("residualWithRegularization: " + math.sqrt(residualWithRegularization))


    // lambda = 4e-5
    // A' = Test Features - trainFeatureMeans
    // A'xComputed
    // A'xComputed + trainLabelMeans
    //Test accuracy should be 45.9%

    test = test.mapPartitions(part=>part(*,::)-trainMeans)

    //println("NormTest (centralized): " + test.normFrobenius())

    val numTestImages = test.numRows().toInt
    val xBroadcast = test.rdd.context.broadcast(x)
    var prediction = test.mapPartitions(mat => mat * xBroadcast.value)
    prediction = prediction.mapPartitions(part=>part(*,::) + trainBMeans)

    val predictionArray = prediction.rdd.map(p=>p.mat).flatMap { p =>
      p.data.grouped(p.rows).toSeq.transpose.map(y => y.toArray)
    }
    val predictedLabels = Utils.topKClassifier(5, predictionArray)
    val errPercent = Utils.getErrPercent(predictedLabels, testLabels, numTestImages)

    //println("Rows of test " + test.numRows())
    //println("Cols of test " + test.numCols())
    println("TestError: " + errPercent)
    */
  }
}