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


object CheckQR extends Logging with Serializable {

  def main(args: Array[String]) {
    if (args.length < 6) {
      println("Usage: CheckQR <master> <data_dir> <parts> <dataset> <numRows> <numCols>")
      System.exit(0)
    }

    val sparkMaster = args(0)
    // Directory that holds the data
    val directory = args(1)
    val parts = args(2).toInt
    //Dataset - currently have lcs, timit, and daisy
    val dataset = args(3).toString
    val numRows = args(4).toInt
    val numCols = args(5).toInt

    println("Running Fusion with ")
    println("master: " + sparkMaster)
    println("directory: " + directory)
    println("parts: " + parts)
    println("dataset: " + dataset)
    println("numRows: " + numRows)
    println("numCols: " + numCols)

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
      case "gaussian-100-10" =>
        trainFilename += "A-Gaussian-100-10/"
      case "gaussian-500-100" =>
        trainFilename += "A-Gaussian-500-100/"
      case "gaussian-1000-10" =>
        trainFilename += "A-Gaussian-1000-10/"
      case "gaussian-10000-10" =>
          trainFilename += "A-Gaussian-10000-10/"
      case "gaussian-16-2" =>
        trainFilename += "A-Gaussian-16-2/"
      case "gaussian-8-1" =>
        trainFilename += "A-Gaussian-8-1"
      case "gaussian-1281167-4001" =>
        trainFilename += "A-Gaussian-1281167-4001"
      case _ =>
        logError("Invalid dataset, " + dataset + " should be in {timit, lcs, daisy}")
        logError("Using daisy")
        trainFilename += "daisy-aPart1-1/"
        bFilename += "daisy-null-labels/"
    }

    /*
    //Create random Gaussian matrix
    val train = RowPartitionedMatrix.createRandomGaussian(sc, numRows, numCols, parts, true).cache()
    train.rdd.count


    //Save matrix
    train.rdd.flatMap(part => MatrixUtils.matrixToRowArray(part.mat)).map {
      x => x.toArray.mkString(",")
    }.saveAsTextFile(directory + "A-Gaussian-500-100")
    */


    val trainRDD = Utils.loadMatrixFromFile(sc, trainFilename, parts).cache()
    trainRDD.count
    var train = RowPartitionedMatrix.fromArray(trainRDD).cache()
    train.rdd.count
    trainRDD.unpersist()


    val (q, r) = new TSQR().qrQR(train)
    //csvwrite(new File("A"), train.collect())
    //csvwrite( new File("Q"), q.collect())
    //csvwrite( new File("R"), r)
    val qr = q.mapPartitions(part => part*r)
    val normA = train.normFrobenius()
    println("norm(A) is " + normA)
    println("norm(A-QR)/norm(A) is " + (train-qr).normFrobenius()/normA)
    val qtq = q.mapPartitions(part=>part.t*part).rdd.map(part=>part.mat).reduce(_+_)
    println("norm(Q^TQ - I) is " + norm((qtq - DenseMatrix.eye[Double](qtq.rows)).toDenseVector))



  }

}
