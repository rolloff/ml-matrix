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


object CheckQR extends Logging with Serializable {

  def main(args: Array[String]) {
    if (args.length < 5) {
      println("Usage: CheckQR <master> <data_dir> <parts> <lambda> <thresh>")
      System.exit(0)
    }

    val sparkMaster = args(0)
    // Directory that holds the data
    val directory = args(1)
    val parts = args(2).toInt
    // Lambda for regularization
    val lambda = args(3).toDouble
    //Threshold for error checks
    val thresh = args(4).toDouble

    println("Running Fusion with ")
    println("master: " + sparkMaster)
    println("directory: " + directory)
    println("parts: " + parts)
    println("lambda: " + lambda)
    println("thresh: " + thresh)

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("CheckQR")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)

    // Daisy filenames
    val daisyTrainFilename = directory + "daisy-aPart1-1/"
    val daisyBFilename = directory + "daisy-null-labels/"

    // LCS filenames
    val lcsTrainFilename = directory + "lcs-aPart1-1/"
    val lcsBFilename = directory + "lcs-null-labels/"

    // load matrix RDDs
    val daisyTrainRDD = Utils.loadMatrixFromFile(sc, daisyTrainFilename, parts)
    val daisyBRDD = Utils.loadMatrixFromFile(sc, daisyBFilename, parts)

    val lcsTrainRDD = Utils.loadMatrixFromFile(sc, lcsTrainFilename, parts)
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


    // Solve for Daisy x using local QR solve
    val localA = daisyTrain.collect()
    val localB = daisyB.collect()
    val reg = DenseMatrix.eye[Double](localA.cols) :* math.sqrt(lambda)

    /*
    //Perform concatenation before QR
    val toSolve = DenseMatrix.vertcat(localA, reg)
    val localQR = qr(toSolve)
    val localQTB = (localQR.q.t * DenseMatrix.vertcat(localB, DenseMatrix.zeros[Double](localA.cols, localB.cols)))
    val localX = localQR.r \ localQTB
    */
    //Perform concatenation after QR
    val localQR = qr(localA)
    val localQTB = (localQR.q.t*localB)
    val localQTBStacked = DenseMatrix.vertcat(localQTB, DenseMatrix.zeros[Double](localA.cols, localB.cols))
    val localRStacked = DenseMatrix.vertcat(localQR.r, reg)
    val localXQR = localRStacked \ localQTBStacked

    //Local Normal Equations
    val ATA = localA.t*localA
    val ATB = localA.t*localB
    val localXNormal = (ATA + (DenseMatrix.eye[Double](ATA.rows):*lambda)) \ ATB

    //Daisy QR results
    val (daisyR, daisyQTB) = new TSQR().returnQRResult(daisyTrain,daisyB)
    val daisyRStacked = DenseMatrix.vertcat(daisyR, DenseMatrix.eye[Double](daisyR.cols):*math.sqrt(lambda))
    val daisyQTBStacked = DenseMatrix.vertcat(daisyQTB, new DenseMatrix[Double](daisyR.cols, daisyQTB.cols))
    val daisyXQR = daisyRStacked \ daisyQTBStacked

    //Daisy Normal Equation results
    val daisyXNormal = new NormalEquations().solveLeastSquaresWithL2(daisyTrain, daisyB, lambda)


    if(Utils.aboutEq(daisyXNormal, localXNormal)){
      println("x from normal paasses")
    }else{
      println("x from normal fails")
    }


    if(Utils.aboutEq(daisyXQR, localXQR)){
      println("x from QR passes")
    }else{
      println("x from QR fails")
    }

    val localQRResidual = Utils.computeResidualNormWithL2(localA, localB, localXQR, lambda)
    val localNormalResidual = Utils.computeResidualNormWithL2(localA, localB, localXNormal, lambda)
    val distributedQRResidual = Utils.computeResidualNormWithL2(daisyTrain, daisyB, daisyXQR, lambda)
    val distributedNormalResidual = Utils.computeResidualNormWithL2(daisyTrain, daisyB, daisyXNormal, lambda)

    println("Local QR Residual is " + localQRResidual)
    println("Local Normal Residual is " + localNormalResidual)
    println("Distributed QR Residual is " + distributedQRResidual)
    println("Distributed Normal Residual is " + distributedNormalResidual)

  }
}
