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

  /* Returns x*/
  def localQR(a1: DenseMatrix[Double],
    a2: DenseMatrix[Double],
     a3: DenseMatrix[Double],
     a4: DenseMatrix[Double],
     b1: DenseMatrix[Double],
     b2: DenseMatrix[Double],
     b3: DenseMatrix[Double],
     b4: DenseMatrix[Double], lambda: Double): DenseMatrix[Double] = {
    val QR1 = qr(a1)
    val QR2 = qr(a2)
    val QR3 = qr(a3)
    val QR4 = qr(a4)

    var R = DenseMatrix.vertcat(QR1.r, QR2.r)
    R = DenseMatrix.vertcat(R, QR3.r)
    R = DenseMatrix.vertcat(R, QR4.r)

    val QTB1 = QR1.q.t*b1
    val QTB2 = QR2.q.t*b2
    val QTB3 = QR3.q.t*b3
    val QTB4 = QR4.q.t*b4

    var QTB = DenseMatrix.vertcat(QTB1, QTB2)
    QTB = DenseMatrix.vertcat(QTB, QTB3)
    QTB = DenseMatrix.vertcat(QTB, QTB4)

    val reg = DenseMatrix.eye[Double](R.cols) :* math.sqrt(lambda)
    val QTBStacked = DenseMatrix.vertcat(QTB, DenseMatrix.zeros[Double](R.cols, b1.cols))
    val RStacked = DenseMatrix.vertcat(R, reg)
    RStacked \ QTBStacked
  }

  def localNormal(a1: DenseMatrix[Double],
    a2: DenseMatrix[Double],
    a3: DenseMatrix[Double],
    a4: DenseMatrix[Double],
    b1: DenseMatrix[Double],
     b2: DenseMatrix[Double],
     b3: DenseMatrix[Double],
     b4: DenseMatrix[Double], lambda: Double): DenseMatrix[Double] = {
    val ATA = a1.t*a1 + a2.t*a2 + a3.t*a3 + a4.t*a4
    val ATB = a1.t*b1 + a2.t*b2 + a3.t*b3 + a4.t*b4
    (ATA + (DenseMatrix.eye[Double](ATA.rows):*lambda)) \ ATB
  }

  def localResidual(x: DenseMatrix[Double],
    a1: DenseMatrix[Double],
    a2: DenseMatrix[Double],
    a3: DenseMatrix[Double],
    a4: DenseMatrix[Double],
    b1: DenseMatrix[Double],
    b2: DenseMatrix[Double],
    b3: DenseMatrix[Double],
    b4: DenseMatrix[Double], lambda: Double): Double = {
    val residSquared = norm((a1*x - b1).toDenseVector)*norm((a1*x-b1).toDenseVector) + norm((a2*x-b2).toDenseVector)*norm((a2*x - b2).toDenseVector) + norm((a3*x-b3).toDenseVector)*norm((a3*x-b3).toDenseVector) + norm((a4*x-b4).toDenseVector)*norm((a4*x-b4).toDenseVector) + lambda*norm(x.toDenseVector)*norm(x.toDenseVector)
    math.sqrt(residSquared)
  }


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


    //Daisy QR results
    val (daisyR, daisyQTB) = new TSQR().returnQRResult(daisyTrain,daisyB)
    val daisyRStacked = DenseMatrix.vertcat(daisyR, DenseMatrix.eye[Double](daisyR.cols):*math.sqrt(lambda))
    val daisyQTBStacked = DenseMatrix.vertcat(daisyQTB, new DenseMatrix[Double](daisyR.cols, daisyQTB.cols))
    val daisyXQR = daisyRStacked \ daisyQTBStacked

    //Daisy Normal Equation results
    val daisyXNormal = new NormalEquations().solveLeastSquaresWithL2(daisyTrain, daisyB, lambda)


    val distributedQRResidual = Utils.computeResidualNormWithL2(daisyTrain, daisyB, daisyXQR, lambda)
    val distributedNormalResidual = Utils.computeResidualNormWithL2(daisyTrain, daisyB, daisyXNormal, lambda)
    println("Distributed QR Residual is " + distributedQRResidual)
    println("Distributed Normal Residual is " + distributedNormalResidual)



    /*
    // Solve for Daisy x using local QR solve
    val localA = daisyTrain.collect()
    val localB = daisyB.collect()
    val reg = DenseMatrix.eye[Double](localA.cols) :* math.sqrt(lambda)


    //Perform concatenation before QR
    val toSolve = DenseMatrix.vertcat(localA, reg)
    val localQR = qr(toSolve)
    val localQTB = (localQR.q.t * DenseMatrix.vertcat(localB, DenseMatrix.zeros[Double](localA.cols, localB.cols)))
    val localX = localQR.r \ localQTB

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
    */

    val numRows = daisyTrain.numRows()
    println("numRows is " + numRows)


    val m = math.floor(numRows/4).toInt
    val a1 = daisyTrain(0 until m, ::).collect()
    val a2 = daisyTrain(m until 2*m, ::).collect()
    val a3 = daisyTrain(2*m until 3*m, ::).collect()
    val a4 = daisyTrain(3*m until numRows.toInt, ::).collect()

    val b1 = daisyB(0 until m, ::).collect()
    val b2 = daisyB(m until 2*m, ::).collect()
    val b3 = daisyB(2*m until 3*m, ::).collect()
    val b4 = daisyB(3*m until numRows.toInt, ::).collect()

    val localXQR = localQR(a1, a2, a3, a4, b1, b2, b3, b4, lambda)
    val localXNormal = localNormal(a1, a2, a3, a4, b1, b2, b3, b4, lambda)


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

    val localQRResidual = localResidual(localXQR, a1, a2, a3, a4, b1, b2, b3, b4, lambda)
    val localNormalResidual = localResidual(localXNormal, a1, a2, a3, a4, b1, b2, b3, b4, lambda)

    println("Local QR Residual is " + localQRResidual)
    println("Local Normal Residual is " + localNormalResidual)


  }
}
