package edu.berkeley.cs.amplab.mlmatrix

import java.io.File
import java.util.concurrent.ThreadLocalRandom

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
    val QR1 = QRUtils.qrQR(a1)
    val QR2 = QRUtils.qrQR(a2)
    val QR3 = QRUtils.qrQR(a3)
    val QR4 = QRUtils.qrQR(a4)

    var R = DenseMatrix.vertcat(QR1._2, QR2._2)
    R = DenseMatrix.vertcat(R, QR3._2)
    R = DenseMatrix.vertcat(R, QR4._2)

    csvwrite(new File("LocalRMatrix-"+ scala.util.Random.nextInt),  R)

    val QTB1 = QR1._1.t*b1
    val QTB2 = QR2._1.t*b2
    val QTB3 = QR3._1.t*b3
    val QTB4 = QR4._1.t*b4

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
    if (args.length < 6) {
      println("Usage: CheckQR <master> <data_dir> <parts> <lambda> <thresh> <dataset>")
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
    //Dataset - currently have lcs, timit, and daisy
    val dataset = args(5).toString

    println("Running Fusion with ")
    println("master: " + sparkMaster)
    println("directory: " + directory)
    println("parts: " + parts)
    println("lambda: " + lambda)
    println("thresh: " + thresh)
    println("dataset: " + dataset)

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("CheckQR")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)

    // Filenames
    var trainFilename = directory
    var bFilename = directory
    dataset.toLowerCase match {
      case "daisy" =>
        trainFilename += "daisy-aPart1-1/"
        bFilename += "daisy-null-labels/"
      case "lcs" =>
        trainFilename += "lcs-aPart1-1/"
        bFilename += "lcs-null-labels/"
      case "timit" =>
        trainFilename += "timit-fft-aPart1-1/"
        bFilename += "timit-fft-null-labels/"
      case _ =>
        logError("Invalid dataset, " + dataset + " should be in {timit, lcs, daisy}")
        logError("Using daisy")
        trainFilename += "daisy-aPart1-1/"
        bFilename += "daisy-null-labels/"
    }


    // load matrix RDDs
    val trainRDD = Utils.loadMatrixFromFile(sc, trainFilename, parts)
    val bRDD = Utils.loadMatrixFromFile(sc, bFilename, parts)
    var trainZipped = trainRDD.zip(bRDD).repartition(parts).cache()

    // Lets cache and assert a few things
    trainZipped.count

    // Create matrices
    var train = RowPartitionedMatrix.fromArray(trainZipped.map(p => p._1)).cache()
    val b = RowPartitionedMatrix.fromArray(trainZipped.map(p => p._2)).cache()

    train.rdd.count
    b.rdd.count
    trainZipped.unpersist()


    // Distributed QR
    val result = new TSQR().returnQRResult(train, b)
    val R = result._1
    csvwrite(new File("DistributedRMatrix-"+ scala.util.Random.nextInt),  R)
    val QTB = result._2
    val RStacked = DenseMatrix.vertcat(R, DenseMatrix.eye[Double](R.cols):*math.sqrt(lambda))
    val QTBStacked = DenseMatrix.vertcat(QTB, new DenseMatrix[Double](R.cols, QTB.cols))
    val XQR = RStacked \ QTBStacked
    csvwrite(new File("DistributedXQR-"+ scala.util.Random.nextInt),  XQR)

    // Distributed Normal Equations
    val XNormal = new NormalEquations().solveLeastSquaresWithL2(train, b, lambda)
    csvwrite(new File("DistributedXNormal-"+ scala.util.Random.nextInt),  XNormal)

    val distributedQRResidual = Utils.computeResidualNormWithL2(train, b, XQR, lambda)
    val distributedNormalResidual = Utils.computeResidualNormWithL2(train, b, XNormal, lambda)
    println("Distributed QR Residual is " + distributedQRResidual)
    println("Distributed Normal Residual is " + distributedNormalResidual)


    val numRows = train.numRows()
    println("numRows is " + numRows)


    // Collect locally into four different matrices to avoid negative java array exception
    val m = math.floor(numRows/4).toInt
    val a1 = train(0 until m, ::).collect()
    val a2 = train(m until 2*m, ::).collect()
    val a3 = train(2*m until 3*m, ::).collect()
    val a4 = train(3*m until numRows.toInt, ::).collect()

    val b1 = b(0 until m, ::).collect()
    val b2 = b(m until 2*m, ::).collect()
    val b3 = b(2*m until 3*m, ::).collect()
    val b4 = b(3*m until numRows.toInt, ::).collect()


    // Local Normal Solve
    val localXNormal = localNormal(a1, a2, a3, a4, b1, b2, b3, b4, lambda)
    csvwrite(new File("LocalXNormal-"+ scala.util.Random.nextInt),  localXNormal)

    val normalRelError = norm(XNormal.toDenseVector - localXNormal.toDenseVector) / norm(localXNormal.toDenseVector)
    println("Relative error between distributed normal solve and local normal solve is " + normalRelError)
    if(normalRelError < thresh) {
      println("x from normal passes")
    }else{
      println("x from normal fails")
    }
    val localNormalResidual = localResidual(localXNormal, a1, a2, a3, a4, b1, b2, b3, b4, lambda)
    println("Local Normal Residual is " + localNormalResidual)


    // Local QR Solve
    val localXQR = localQR(a1, a2, a3, a4, b1, b2, b3, b4, lambda)
    csvwrite(new File("LocalXQR-"+ scala.util.Random.nextInt),  localXQR)


    val qrRelError = norm(XQR.toDenseVector - localXQR.toDenseVector) / norm(localXQR.toDenseVector)
    println("Relative error between distributed qr solve and local qr solve is " + qrRelError)
    if(qrRelError < thresh){
      println("x from QR passes")
    }else{
      println("x from QR fails")
    }

    val localQRResidual = localResidual(localXQR, a1, a2, a3, a4, b1, b2, b3, b4, lambda)
    println("Local QR Residual is " + localQRResidual)



  }
}
