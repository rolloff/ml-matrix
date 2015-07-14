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
    println("lambda"  + lambda)

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("CheckSolvers")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)

    // Filenames
    var trainFilename = directory
    var bFilename = directory
    dataset.toLowerCase match {
      case "imagenet-fv-4k" =>
        trainFilename += "imagenenet-fv-4k/trainFeatures-all.csv"
        bFilename += "imagenet-fv-4k/trainLabels-all.csv"
      case _ =>
        logError("Invalid dataset")
        logError("Using imagenet-fv-4k")
        trainFilename += "imagenenet-fv-4k/trainFeatures-all.csv"
        bFilename += "imagenet-fv-4k/trainLabels-all.csv"
    }

    val trainRDD = Utils.loadMatrixFromFile(sc, trainFilename, parts)
    val bClasses = sc.textFile(bFilename, parts).map(line => line.toInt)
    // Create matrix of +1/-1s given class labels
    // Assume classId in [1,1000]
    println("The maximum class label is " + bClasses.max())
    println("The minimum class label is " + bClasses.min())
    val bRDD = bClasses.map { classId =>
      val classes = Array.fill[Double](1000)(-1)
      classes(classId-1) = 1
      classes
    }
    val trainZipped = trainRDD.zip(bRDD).repartition(parts).cache()

    trainZipped.count
    val train = RowPartitionedMatrix.fromArray(trainZipped.map(p=>p._1)).cache()
    val b = RowPartitionedMatrix.fromArray(trainZipped.map(p=>p._2)).cache()

    train.rdd.count
    b.rdd.count
    trainZipped.unpersist()

    val x: DenseMatrix[Double] = solver.toLowerCase match {
      case "normal" =>
        new NormalEquations().solveLeastSquaresWithL2(train, b, lambda)
      case "tsqr" =>
        new TSQR().solveLeastSquaresWithL2(train, b, lambda)
      case _ =>
        logError("Invalid Solver ")
        logError("Using Normal Equations")
        new NormalEquations().solveLeastSquaresWithL2(train, b, lambda)
    }

    // Record normA, normB, normX, norm(AX-B), norm(AX-B) + lambda*norm(X)
    val residual = Utils.computeResidualNormWithL2(train, b, x, lambda)
    val normA = train.normFrobenius()
    val normB = b.normFrobenius()
    val normX = norm(x.toDenseVector)
    val residualWithRegularization = residual*residual+lambda*normX*normX
    println("Residual: " + residual)
    println("normA: " + normA)
    println("normB: " + normB)
    println("normX: " + normX)
    println("residualWithRegularization: " + residualWithRegularization)
  }
}
