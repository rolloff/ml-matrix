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


object CheckQREight extends Logging with Serializable {

  /* Returns x*/
  def localQR(a1: DenseMatrix[Double],
    a2: DenseMatrix[Double],
     a3: DenseMatrix[Double],
     a4: DenseMatrix[Double],
     a5: DenseMatrix[Double],
     a6: DenseMatrix[Double],
     a7: DenseMatrix[Double],
     a8: DenseMatrix[Double],
     b1: DenseMatrix[Double],
     b2: DenseMatrix[Double],
     b3: DenseMatrix[Double],
     b4: DenseMatrix[Double],
     b5: DenseMatrix[Double],
     b6: DenseMatrix[Double],
     b7: DenseMatrix[Double],
     b8: DenseMatrix[Double], lambda: Double): DenseMatrix[Double] = {
    val QR1 = qr(a1)
    val QR2 = qr(a2)
    val QR3 = qr(a3)
    val QR4 = qr(a4)
    val QR5 = qr(a5)
    val QR6 = qr(a6)
    val QR7 = qr(a7)
    val QR8 = qr(a8)

    var R = DenseMatrix.vertcat(QR1.r, QR2.r)
    R = DenseMatrix.vertcat(R, QR3.r)
    R = DenseMatrix.vertcat(R, QR4.r)
    R = DenseMatrix.vertcat(R, QR5.r)
    R = DenseMatrix.vertcat(R, QR6.r)
    R = DenseMatrix.vertcat(R, QR7.r)
    R = DenseMatrix.vertcat(R, QR8.r)


    val QTB1 = QR1.q.t*b1
    val QTB2 = QR2.q.t*b2
    val QTB3 = QR3.q.t*b3
    val QTB4 = QR4.q.t*b4
    val QTB5 = QR5.q.t*b5
    val QTB6 = QR6.q.t*b6
    val QTB7 = QR7.q.t*b7
    val QTB8 = QR8.q.t*b8

    var QTB = DenseMatrix.vertcat(QTB1, QTB2)
    QTB = DenseMatrix.vertcat(QTB, QTB3)
    QTB = DenseMatrix.vertcat(QTB, QTB4)
    QTB = DenseMatrix.vertcat(QTB, QTB5)
    QTB = DenseMatrix.vertcat(QTB, QTB6)
    QTB = DenseMatrix.vertcat(QTB, QTB7)
    QTB = DenseMatrix.vertcat(QTB, QTB8)


    val reg = DenseMatrix.eye[Double](R.cols) :* math.sqrt(lambda)
    val QTBStacked = DenseMatrix.vertcat(QTB, DenseMatrix.zeros[Double](R.cols, b1.cols))
    val RStacked = DenseMatrix.vertcat(R, reg)
    RStacked \ QTBStacked
  }

  def localNormal(a1: DenseMatrix[Double],
    a2: DenseMatrix[Double],
     a3: DenseMatrix[Double],
     a4: DenseMatrix[Double],
     a5: DenseMatrix[Double],
     a6: DenseMatrix[Double],
     a7: DenseMatrix[Double],
     a8: DenseMatrix[Double],
     b1: DenseMatrix[Double],
     b2: DenseMatrix[Double],
     b3: DenseMatrix[Double],
     b4: DenseMatrix[Double],
     b5: DenseMatrix[Double],
     b6: DenseMatrix[Double],
     b7: DenseMatrix[Double],
     b8: DenseMatrix[Double], lambda: Double): DenseMatrix[Double] = {

    val ATA = a1.t*a1 + a2.t*a2 + a3.t*a3 + a4.t*a4 + a5.t*a5 + a6.t*a6 + a7.t*a7 + a8.t*a8
    val ATB = a1.t*b1 + a2.t*b2 + a3.t*b3 + a4.t*b4 + a5.t*b5 + a6.t*b6 + a7.t*b7 + a8.t*b8
    (ATA + (DenseMatrix.eye[Double](ATA.rows):*lambda)) \ ATB
  }

  def localResidual(x: DenseMatrix[Double],
     a1: DenseMatrix[Double],
     a2: DenseMatrix[Double],
     a3: DenseMatrix[Double],
     a4: DenseMatrix[Double],
     a5: DenseMatrix[Double],
     a6: DenseMatrix[Double],
     a7: DenseMatrix[Double],
     a8: DenseMatrix[Double],
     b1: DenseMatrix[Double],
     b2: DenseMatrix[Double],
     b3: DenseMatrix[Double],
     b4: DenseMatrix[Double],
     b5: DenseMatrix[Double],
     b6: DenseMatrix[Double],
     b7: DenseMatrix[Double],
     b8: DenseMatrix[Double], lambda: Double): Double = {

    val residSquared = norm((a1*x - b1).toDenseVector)*norm((a1*x-b1).toDenseVector)
    + norm((a2*x-b2).toDenseVector)*norm((a2*x - b2).toDenseVector) +
    norm((a3*x-b3).toDenseVector)*norm((a3*x-b3).toDenseVector) +
    norm((a4*x-b4).toDenseVector)*norm((a4*x-b4).toDenseVector) +
    norm((a5*x-b5).toDenseVector)*norm((a5*x-b5).toDenseVector) +
    norm((a6*x-b6).toDenseVector)*norm((a6*x-b6).toDenseVector) +
    norm((a7*x-b7).toDenseVector)*norm((a7*x-b7).toDenseVector) +
    norm((a8*x-b8).toDenseVector)*norm((a8*x-b8).toDenseVector) +
    lambda*norm(x.toDenseVector)*norm(x.toDenseVector)
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


    val m = math.floor(numRows/8).toInt

    val a1 = daisyTrain(0 until m, ::).collect()
    val a2 = daisyTrain(m until 2*m, ::).collect()
    val a3 = daisyTrain(2*m until 3*m, ::).collect()
    val a4 = daisyTrain(3*m until 4*m, ::).collect()
    val a5 = daisyTrain(4*m until 5*m, ::).collect()
    val a6 = daisyTrain(5*m until 6*m, ::).collect()
    val a7 = daisyTrain(6*m until 7*m, ::).collect()
    val a8 = daisyTrain(7*m until numRows.toInt, ::).collect()


    val b1 = daisyTrain(0 until m, ::).collect()
    val b2 = daisyTrain(m until 2*m, ::).collect()
    val b3 = daisyTrain(2*m until 3*m, ::).collect()
    val b4 = daisyTrain(3*m until 4*m, ::).collect()
    val b5 = daisyTrain(4*m until 5*m, ::).collect()
    val b6 = daisyTrain(5*m until 6*m, ::).collect()
    val b7 = daisyTrain(6*m until 7*m, ::).collect()
    val b8 = daisyTrain(7*m until numRows.toInt, ::).collect()

    val localXQR = localQR(a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, lambda)
    val localXNormal = localNormal(a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, lambda)


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

    val localQRResidual = localResidual(localXQR, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, lambda)
    val localNormalResidual = localResidual(localXNormal, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8, lambda)

    println("Local QR Residual is " + localQRResidual)
    println("Local Normal Residual is " + localNormalResidual)


  }
}
