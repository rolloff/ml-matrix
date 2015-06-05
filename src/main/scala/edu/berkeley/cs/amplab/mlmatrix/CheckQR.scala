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
     b4: DenseMatrix[Double], lambda: Double): (DenseMatrix[Double], DenseMatrix[Double]) = {
    println("running localQR1")
    val QR1 = QRUtils.qrQR(a1)
    println("running localQR2")
    val QR2 = QRUtils.qrQR(a2)
    println("running localQR3")
    val QR3 = QRUtils.qrQR(a3)
    println("running localQR4")
    val QR4 = QRUtils.qrQR(a4)

    var R = DenseMatrix.vertcat(QR1._2, QR2._2)
    R = DenseMatrix.vertcat(R, QR3._2)
    R = DenseMatrix.vertcat(R, QR4._2)

    println("Frobenius norm of B is " + math.sqrt(
      norm(b1.toDenseVector)*norm(b1.toDenseVector)
      + norm(b2.toDenseVector)*norm(b2.toDenseVector)
      + norm(b3.toDenseVector)*norm(b3.toDenseVector)
      +norm(b4.toDenseVector)*norm(b4.toDenseVector))
    )

    println("Norm of b1: " + norm(b1.toDenseVector))
    val QTB1 = QR1._1.t*b1
    println("Norm of QTB1: " + norm(QTB1.toDenseVector))
    println("Norm of b2: " +  norm(b2.toDenseVector))
    val QTB2 = QR2._1.t*b2
    println("Norm of QTB2: " + norm(QTB2.toDenseVector))
    println("Norm of b3: " + norm(b3.toDenseVector))
    val QTB3 = QR3._1.t*b3
    println("Norm of QTB3: " + norm(QTB3.toDenseVector))
    println("Norm of b4: " + norm(b4.toDenseVector))
    val QTB4 = QR4._1.t*b4
    println("Norm of QTB4: " + norm(QTB4.toDenseVector))

    var QTB = DenseMatrix.vertcat(QTB1, QTB2)
    QTB = DenseMatrix.vertcat(QTB, QTB3)
    QTB = DenseMatrix.vertcat(QTB, QTB4)

    println("Norm of vertically concatenated QTB: " + norm(QTB.toDenseVector))
    csvwrite(new File("LocalQTB_Before_Final_Reduction-" + scala.util.Random.nextInt), QTB)

    // Final QR reduction
    val QR5 = QRUtils.qrQR(R)
    R = QR5._2
    QTB = QR5._1.t*QTB
    println("Norm of final QTB: " + norm(QTB.toDenseVector))

    csvwrite(new File("LocalQTBFinal-" + scala.util.Random.nextInt), QTB)
    csvwrite(new File("LocalRMatrix-"+ scala.util.Random.nextInt),  R)
    val reg = DenseMatrix.eye[Double](R.cols) :* math.sqrt(lambda)
    val QTBStacked = DenseMatrix.vertcat(QTB, DenseMatrix.zeros[Double](R.cols, b1.cols))
    val RStacked = DenseMatrix.vertcat(R, reg)
    (QTBStacked, RStacked)
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
    //val bRDD = Utils.loadMatrixFromFile(sc, bFilename, parts)
    //var trainZipped = trainRDD.zip(bRDD).repartition(parts).cache()

    // Lets cache and assert a few things
    //trainZipped.count

    // Create matrices
    //var train = RowPartitionedMatrix.fromArray(trainZipped.map(p => p._1)).cache()
    //val b = RowPartitionedMatrix.fromArray(trainZipped.map(p => p._2)).cache()
    var train = RowPartitionedMatrix.fromArray(trainRDD)

    train.rdd.count
    //b.rdd.count
    //trainZipped.unpersist()

    val rowSizes = train.rdd.map{ x => x.mat.rows}.collect()
    println("Row Sizes : " + rowSizes.mkString(","))


    //Measuring norm(A-QR)/norm(A) and norm(QTQ-I)/norm(Q)
    val (q, r) = new TSQR().qrQR(train)
    val qr = q.mapPartitions{ part => part*r}
    println("norm(A-QR)/norm(A) is " + (train-qr).normFrobenius()/train.normFrobenius())
    val qtq = q.mapPartitions(part=>part.t*part).rdd.map(part=>part.mat).reduce(_+_)
    println("norm(Q^TQ - I) is " + norm((qtq - DenseMatrix.eye[Double](qtq.rows)).toDenseVector))
  }

    // Distributed QR
    /*
    val result = new TSQR().returnQRResult(train, b)
    val R = result._1
    csvwrite(new File("DistributedR-"+ parts ),  R)
    val QTB = result._2
    csvwrite(new File("DistributedQTB-" +parts), QTB)
    val RStacked = DenseMatrix.vertcat(R, DenseMatrix.eye[Double](R.cols):*math.sqrt(lambda))
    val QTBStacked = DenseMatrix.vertcat(QTB, new DenseMatrix[Double](R.cols, QTB.cols))
    val XQR = RStacked \ QTBStacked
    csvwrite(new File("DistributedX-"+ parts), XQR)
    val distributedQRResidual = RStacked*XQR - QTBStacked

    csvwrite(new File("DistributedResidual-"+ parts),  distributedQRResidual)
    val normDistributedQRResidual = Utils.computeResidualNormWithL2(train, b, XQR, lambda)


    println("Distributed Norm of Rx-QTB" + norm(distributedQRResidual.toDenseVector))
    println("Distributed Norm of Ax-b " + normDistributedQRResidual)

    */


    /*
    // Collect locally into four different matrices to avoid negative java array exception
    val numRows = train.numRows()
    val m = math.floor(numRows/8).toInt
    var a1 = train(0 until m, ::).collect()
    var a2 = train(m until 2*m, ::).collect()
    var a3 = train(2*m until 3*m, ::).collect()
    var a4 = train(3*m until 4*m, ::).collect()
    var a5 = train(4*m until 5*m, ::).collect()
    var a6 = train(5*m until 6*m, ::).collect()
    var a7 = train(6*m until 7*m, ::).collect()
    var a8 = train(7*m until numRows.toInt, ::).collect()

    println("FINISHED A COLLECTS")

    var b1 = b(0 until m, ::).collect()
    var b2 = b(m until 2*m, ::).collect()
    var b3 = b(2*m until 3*m, ::).collect()
    var b4 = b(3*m until 4*m, ::).collect()
    var b5 = b(4*m until 5*m, ::).collect()
    var b6 = b(5*m until 6*m, ::).collect()
    var b7 = b(6*m until 7*m, ::).collect()
    var b8 = b(7*m until numRows.toInt, ::).collect()

    println("FINISHED B COLLECTS")

    a1 = DenseMatrix.vertcat(a1, a2)
    a2 = DenseMatrix.vertcat(a3, a4)
    a3 = DenseMatrix.vertcat(a5, a6)
    a4 = DenseMatrix.vertcat(a7, a8)

    println("FINISHED A VERTCATS")

    b1 = DenseMatrix.vertcat(b1, b2)
    b2 = DenseMatrix.vertcat(b3, b4)
    b3 = DenseMatrix.vertcat(b5, b6)
    b4 = DenseMatrix.vertcat(b7, b8)

    println("FINISHED B VERTCATS")

    // Local QR Solve
    val localQRResult = localQR(a1, a2, a3, a4, b1, b2, b3, b4, lambda)
    val localX = localQRResult._2 \localQRResult._1
    csvwrite(new File("LocalX-"+scala.util.Random.nextInt), localX)
    val localQRResidual = localQRResult._2*localX - localQRResult._1

    csvwrite(new File("LocalQRResidual-"+ scala.util.Random.nextInt),  localQRResidual)

    val normLocalQRResidual = localResidual(localX, a1, a2, a3, a4, b1, b2, b3, b4, lambda)
    println("Norm of local QR Residual is " + normLocalQRResidual)
    */


    //Difference
    //println("Norm of residual differences is " + norm((localQRResidual-distributedQRResidual).toDenseVector))


}
