package edu.berkeley.cs.amplab.mlmatrix

import java.io.File
import java.util.concurrent.ThreadLocalRandom
import scala.collection.mutable.ArrayBuffer
import scala.util.Random

import breeze.linalg._

import edu.berkeley.cs.amplab.mlmatrix.util.QRUtils
import edu.berkeley.cs.amplab.mlmatrix.util.Utils

import org.apache.spark.Accumulator
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.TaskContext

class TSQR extends RowPartitionedSolver with Logging with Serializable {

  def qrR(mat: RowPartitionedMatrix): DenseMatrix[Double] = {
    val localQR = mat.rdd.context.accumulator(0.0, "Time taken for Local QR")

    val qrTree = mat.rdd.map { part =>
      if (part.mat.rows < part.mat.cols) {
        part.mat
      } else {
        val begin = System.nanoTime
        val r = QRUtils.qrR(part.mat)
        localQR += ((System.nanoTime - begin) / 1000000)
        r
      }
    }
    val depth = math.ceil(math.log(mat.rdd.partitions.size)/math.log(2)).toInt
    Utils.treeReduce(qrTree, reduceQR(localQR, _ : DenseMatrix[Double], _ : DenseMatrix[Double]), depth=depth)
  }

  private def reduceQR(acc: Accumulator[Double], a: DenseMatrix[Double], b: DenseMatrix[Double]): DenseMatrix[Double] = {
    val begin = System.nanoTime
    val out = QRUtils.qrR(DenseMatrix.vertcat(a, b), false)
    acc += ((System.nanoTime - begin) / 1e6)
    out
  }

  def qrQR(mat: RowPartitionedMatrix): (RowPartitionedMatrix, DenseMatrix[Double]) = {
    // First step run TSQR, get YTR tree
    val (qrTree, r) = qrYTR(mat)

    /*
    //Debug qrTree. qrTree is a Seq[(Int, RDD[(Int, (DenseMatrix[Double], Array[Double], DenseMatrix[Double]))])]
    qrTree.map { part: (Int, RDD[(Int, (DenseMatrix[Double], Array[Double], DenseMatrix[Double]))]) =>
      val treeLevel: Int = part._1
      val treeRDD: RDD[(Int, (DenseMatrix[Double], Array[Double], DenseMatrix[Double]))] = part._2
      println("treeLevel is " + treeLevel + " with tree RDD of size " + treeRDD.partitions.size)
      val info = treeRDD.map { p =>
        val partId: Int = p._1
        val y: DenseMatrix[Double] = p._2._1
        val t: Array[Double] = p._2._2
        csvwrite(new File("y-"+ treeLevel +"-" + partId), y)
        csvwrite(new File("t-" + treeLevel + "-" + partId), DenseMatrix(t))
        (partId, y, t)
      }.collect()
    }
    */


    var curTreeIdx = qrTree.size - 1
    val lastIdx = qrTree.size - 1
    //println("About to start constructing Q with curTreeIdx as " + lastIdx + "and a qrTree of size " + qrTree.size)

    // Now construct Q by going up the tree
    var qrRevTree: RDD[(Int, DenseMatrix[Double])] = qrTree(lastIdx)._2.map { part =>
      val y = part._2._1
      val t = part._2._2
      val qIn = new DenseMatrix[Double](y.rows, y.cols)
      for (i <- 0 until y.cols) {
        qIn(i, i) =  1.0
      }
      val applyQResult = QRUtils.applyQ(y, t, qIn, transpose=false)
      //val treeLevel: Int = qrTree(lastIdx)._1
      val partId: Int = part._1
      //println("LastIdx is " + lastIdx)
      //println("About to write out Q with treeLevel " + treeLevel + " and partId " + partId)
      //csvwrite(new File("Q-" + treeLevel + "-" + partId), applyQResult)

      (partId, applyQResult)
    }.flatMap { x =>
      val nrows = x._2.rows
      Iterator((x._1 * 2, x._2),
               (x._1 * 2 + 1, x._2))
    }
    qrRevTree.cache()
    qrRevTree.count()

    //val prevTree: RDD[(Int, DenseMatrix[Double])] = qrRevTree
    //prevTree.count()
    //println("The size of prevTree is "+ prevTree.partitions.size)


    while (curTreeIdx > 0)
    {
      val whileLoopTreeIdx = curTreeIdx - 1
      curTreeIdx = whileLoopTreeIdx
      //val treeLevel: Int = qrTree(whileLoopTreeIdx)._1
      //println("treeLevel is " + treeLevel)
      val whileLoopPrevTree = qrRevTree
      //prevTree = qrRevTree

      if (whileLoopTreeIdx > 0) {
        //println("With two partitions we should not end up here")
        val nextNumParts = qrTree(whileLoopTreeIdx - 1)._1
        qrRevTree = qrTree(whileLoopTreeIdx)._2.join(whileLoopPrevTree).flatMap { part =>
          val y = part._2._1._1
          val t = part._2._1._2

          val qPart = if (part._1 % 2 == 0) {
            val e = math.min(y.rows, y.cols)
            part._2._2(0 until e, ::)
          } else {
            val numRows = math.min(y.rows, y.cols)
            val s = part._2._2.rows - numRows
            part._2._2(s until part._2._2.rows, ::)
          }

          if (part._1 * 2 + 1 < nextNumParts) {
            val qOut = QRUtils.applyQ(y, t, qPart, transpose=false)
            val nrows = qOut.rows
            Iterator((part._1 * 2, qOut),
                     (part._1 * 2 + 1, qOut))
          } else {
            Iterator((part._1 * 2, qPart))
          }
        }
        //qrRevTree.count()
      } else {
        //println("We should go here immediately with 2 partitions ")
        //println("whileLoopPrevTree has size " + whileLoopPrevTree.count() + " partitions: " + whileLoopPrevTree.partitions.size)
        //println("qrTree(whileLoopTreeIdx)._2 has size " + qrTree(whileLoopTreeIdx)._2.count() + " partitions: " + qrTree(whileLoopTreeIdx)._2.partitions.size)
        qrRevTree = qrTree(whileLoopTreeIdx)._2.join(whileLoopPrevTree).map { part =>
          val partId: Int = part._1
          //println("Inside the join with partId " + partId)
          val y = part._2._1._1
          val t = part._2._1._2
          val qPart = if (part._1 % 2 == 0) {
            val e = math.min(y.rows, y.cols)
            part._2._2(0 until e, ::)
          } else {
            val numRows = math.min(y.rows, y.cols)
            val s = part._2._2.rows - numRows
            part._2._2(s until part._2._2.rows, ::)
          }
          val applyQResult = QRUtils.applyQ(y, t, qPart, transpose=false)
          //csvwrite(new File("Q-" + treeLevel + "-" + partId + "-" + System.currentTimeMillis / 1000), applyQResult)
          (partId, applyQResult)
        }
        //qrRevTree.cache()
        //println("Final size of qrRevTree is " + qrRevTree.count())
      }
      qrRevTree.cache()
      qrRevTree.count()
    }

    (RowPartitionedMatrix.fromMatrix(qrRevTree.map(x => x._2)), r)
  }

  private def qrYTR(mat: RowPartitionedMatrix):
      (Seq[(Int, RDD[(Int, (DenseMatrix[Double], Array[Double], DenseMatrix[Double]))])],
        DenseMatrix[Double]) = {
    val qrTreeSeq = new ArrayBuffer[(Int, RDD[(Int, (DenseMatrix[Double], Array[Double], DenseMatrix[Double]))])]

    val matPartInfo: Map[Int, Array[RowPartitionInfo]] = mat.getPartitionInfo
    val matPartInfoBroadcast = mat.rdd.context.broadcast(matPartInfo)

    var qrTree: RDD[(Int, (DenseMatrix[Double], Array[Double], DenseMatrix[Double]))] = mat.rdd.mapPartitionsWithIndex { case (part: Int, iter: Iterator[RowPartition]) =>
      if (matPartInfoBroadcast.value.contains(part) && !iter.isEmpty) {
        val partBlockIds: Array[Int] = matPartInfoBroadcast.value(part).sortBy(x=> x.blockId).map(x => x.blockId)
        val partBlockIdsIterator: Iterator[Int] = partBlockIds.iterator
        //Does each RowPartition correspond to one blockId? How can zip handle this?
        iter.zip(partBlockIds.iterator).map { case (lm: RowPartition, bi: Int) =>
          if (lm.mat.rows < lm.mat.cols) {
            (
              bi,
              (new DenseMatrix[Double](lm.mat.rows, lm.mat.cols),
               new Array[Double](lm.mat.rows),
              lm.mat)
            )
          } else {
            val qrResult = QRUtils.qrYTR(lm.mat)
            (bi, qrResult)
          }
        }
      } else {
        Iterator()
      }
    }

    var numParts = matPartInfo.flatMap(x => x._2.map(y => y.blockId)).size
    qrTreeSeq.append((numParts, qrTree))

    while (numParts > 1) {
      qrTree = qrTree.map(x => ((x._1/2.0).toInt, (x._1, x._2))).reduceByKey(
        numPartitions=math.ceil(numParts/2.0).toInt,
        func=reduceYTR(_, _)).map(x => (x._1, x._2._2))
      numParts = math.ceil(numParts/2.0).toInt
      qrTreeSeq.append((numParts, qrTree))
    }
    val r = qrTree.map(x => x._2._3).collect()(0)
    (qrTreeSeq, r)
  }

  private def reduceYTR(
      a: (Int, (DenseMatrix[Double], Array[Double], DenseMatrix[Double])),
      b: (Int, (DenseMatrix[Double], Array[Double], DenseMatrix[Double])))
    : (Int, (DenseMatrix[Double], Array[Double], DenseMatrix[Double])) = {
    // Stack the lower id above the higher id
    if (a._1 < b._1) {
      (a._1, QRUtils.qrYTR(DenseMatrix.vertcat(a._2._3, b._2._3)))
    } else {
      (b._1, QRUtils.qrYTR(DenseMatrix.vertcat(b._2._3, a._2._3)))
    }
  }

  // From http://math.stackexchange.com/questions/299481/qr-factorization-for-ridge-regression
  // To solve QR with L2, we need to factorize \pmatrix{ A \\ \Gamma}
  // i.e. A and \Gamma stacked vertically, where \Gamma is a nxn Matrix.
  // To do this we first use TSQR on A and then locally stack \Gamma below and recompute QR.
  def solveLeastSquaresWithManyL2(
      A: RowPartitionedMatrix,
      b: RowPartitionedMatrix,
      lambdas: Array[Double]): Seq[DenseMatrix[Double]] = {
    val matrixParts = A.rdd.zip(b.rdd).map(x => (x._1.mat, x._2.mat))
    val localQR = A.rdd.context.accumulator(0.0, "Time taken for Local QR Solve")

    val qrTree = matrixParts.map { part =>
      val (aPart, bPart) = part
      if (aPart.rows < aPart.cols) {
        (aPart, bPart)
      } else {
        val begin = System.nanoTime
        val out = QRUtils.qrSolve(aPart, bPart)
        localQR += ((System.nanoTime - begin) / 1000000)
        out
      }
    }

    val depth = math.ceil(math.log(A.rdd.partitions.size)/math.log(2)).toInt
    val qrResult = Utils.treeReduce(qrTree,
      reduceQRSolve(
        localQR,
        _: (DenseMatrix[Double], DenseMatrix[Double]),
        _: (DenseMatrix[Double], DenseMatrix[Double])),
      depth=depth)

    val results = lambdas.map { lambda =>
      // We only have one partition right now
      val (rFinal, bFinal) = qrResult
      val out = if (lambda == 0.0) {
        rFinal \ bFinal
      } else {
        val lambdaRB = (DenseMatrix.eye[Double](rFinal.cols) :* math.sqrt(lambda),
          new DenseMatrix[Double](rFinal.cols, bFinal.cols))
        val reduced = reduceQRSolve(localQR, (rFinal, bFinal), lambdaRB)
        // reduced._1 is stacked R and sqrt(lambda)*I
        // look up breeze method to save out to csv

        //csvwrite(new File("RMatrix-"+ scala.util.Random.nextInt),  reduced._1)
        // csvwrite(new File("RMatrix-"+ scala.util.Random.nextInt),  rFinal)
        //println("Diagonal elements of stacked R: " + diag(reduced._1).toArray.mkString(" "))
        reduced._1 \ reduced._2
      }
      out
    }
    results
  }

  /*Exactly model situation in solveLeastSquaresWithManyL2 to generate R to pass to CheckQR*/
  def returnQRResult(
    A: RowPartitionedMatrix,
    b: RowPartitionedMatrix): (DenseMatrix[Double], DenseMatrix[Double]) = {
  val matrixParts = A.rdd.zip(b.rdd).map(x => (x._1.mat, x._2.mat))
  val localQR = A.rdd.context.accumulator(0.0, "Time taken for Local QR Solve")

  val qrTree = matrixParts.map { part =>
    val (aPart, bPart) = part
    if (aPart.rows < aPart.cols) {
      (aPart, bPart)
    } else {
      val begin = System.nanoTime
      val out = QRUtils.qrSolve(aPart, bPart)
      localQR += ((System.nanoTime - begin) / 1000000)
      //val stageId = TaskContext.get.stageId
      //val partitionId = TaskContext.get.partitionId
      //println("2-norm of R inside QR tree is " + norm(out._1.toDenseVector) + " at time " + localQR +
      //        "stage " + stageId + " partition " + partitionId)
      out
    }
  }

  val depth = math.ceil(math.log(A.rdd.partitions.size)/math.log(2)).toInt
  val qrResult = Utils.treeReduce(qrTree,
    reduceQRSolve(
      localQR,
      _: (DenseMatrix[Double], DenseMatrix[Double]),
      _: (DenseMatrix[Double], DenseMatrix[Double])),
    depth=depth)
  qrResult
  }

  private def reduceQRSolve(
      acc: Accumulator[Double],
      a: (DenseMatrix[Double], DenseMatrix[Double]),
      b: (DenseMatrix[Double], DenseMatrix[Double])): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val begin = System.nanoTime
    val out = QRUtils.qrSolve(DenseMatrix.vertcat(a._1, b._1),
      DenseMatrix.vertcat(a._2, b._2))
    acc += ((System.nanoTime - begin) / 1e6)
    //val stageId = TaskContext.get.stageId
    //val partitionId = TaskContext.get.partitionId
    //println("2-norm of R inside reduceQR is " + norm(out._1.toDenseVector) +" at time " + acc + " " +
      //      "stage " + stageId + " partition " + partitionId)
    out
  }

  def solveManyLeastSquaresWithL2(
      A: RowPartitionedMatrix,
      b: RDD[Seq[DenseMatrix[Double]]],
      lambdas: Array[Double]): Seq[DenseMatrix[Double]] = {

    val localQR = A.rdd.context.accumulator(0.0, "Time taken for Local QR Solves")

    val matrixParts = A.rdd.zip(b).map { x =>
      (x._1.mat, x._2)
    }

    val qrTree = matrixParts.map { part =>
      val (aPart, bParts) = part

      if (aPart.rows < aPart.cols) {
        (aPart, bParts)
      } else {
        QRUtils.qrSolveMany(aPart, bParts)
      }
    }

    val qrResult = Utils.treeReduce(qrTree, reduceQRSolveMany,
      depth=math.ceil(math.log(A.rdd.partitions.size)/math.log(2)).toInt)

    val rFinal = qrResult._1

    val results = lambdas.zip(qrResult._2).map { case (lambda, bFinal) =>
      // We only have one partition right now
      val out = if (lambda == 0.0) {
        rFinal \ bFinal
      } else {
        val lambdaRB = (DenseMatrix.eye[Double](rFinal.cols) :* math.sqrt(lambda),
          new DenseMatrix[Double](rFinal.cols, bFinal.cols))
        val reduced = reduceQRSolve(localQR, (rFinal, bFinal), lambdaRB)
        //csvwrite(new File("RMatrix-"+ scala.util.Random.nextInt),  rFinal)
        reduced._1 \ reduced._2
      }
      out
    }
    results
  }

  private def reduceQRSolveMany(
      a: (DenseMatrix[Double], Seq[DenseMatrix[Double]]),
      b: (DenseMatrix[Double], Seq[DenseMatrix[Double]])):
        (DenseMatrix[Double], Seq[DenseMatrix[Double]]) = {
    QRUtils.qrSolveMany(DenseMatrix.vertcat(a._1, b._1),
      a._2.zip(b._2).map(x => DenseMatrix.vertcat(x._1, x._2)))
  }
}

object TSQR extends Logging {

  def main(args: Array[String]) {
    if (args.length < 5) {
      println("Usage: TSQR <master> <numRows> <numCols> <numParts> <numClasses>")
      System.exit(0)
    }

    val sparkMaster = args(0)
    val numRows = args(1).toInt
    val numCols = args(2).toInt
    val numParts = args(3).toInt
    val numClasses = args(4).toInt

    val conf = new SparkConf()
      .setMaster(sparkMaster)
      .setAppName("TSQR")
      .setJars(SparkContext.jarOfClass(this.getClass).toSeq)
    val sc = new SparkContext(conf)

    Thread.sleep(5000)

    val rowsPerPart = numRows / numParts
    val matrixParts = sc.parallelize(1 to numParts, numParts).mapPartitions { part =>
      val data = new Array[Double](rowsPerPart * numCols)
      var i = 0
      while (i < rowsPerPart * numCols) {
        data(i) = ThreadLocalRandom.current().nextGaussian()
        i = i + 1
      }
      val mat = new DenseMatrix[Double](rowsPerPart, numCols, data)
      Iterator(mat)
    }
    matrixParts.cache().count()

    var begin = System.nanoTime()
    val A = RowPartitionedMatrix.fromMatrix(matrixParts)
    val R = new TSQR().qrR(A)
    var end = System.nanoTime()
    logInfo("Random TSQR of " + numRows + "x" + numCols + " took " + (end - begin)/1e6 + "ms")

    var c = readChar
    sc.stop()
  }

}
