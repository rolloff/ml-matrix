package edu.berkeley.cs.amplab.mlmatrix.util

import scala.reflect.ClassTag
import scala.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.HashPartitioner
import org.apache.spark.Partitioner
import org.apache.spark.rdd.RDD
import org.apache.spark.rdd.ShuffledRDD

import edu.berkeley.cs.amplab.mlmatrix.RowPartitionedMatrix

import breeze.linalg._

object Utils {

  /**
   * Deep copy a Breeze matrix
   */
  def cloneMatrix(in: DenseMatrix[Double]) = {
    in.copy
  }

  def decomposeLowerUpper(A: DenseMatrix[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val L = new DenseMatrix[Double](A.rows, min(A.rows, A.cols))
    val U = new DenseMatrix[Double](min(A.cols, A.rows), A.cols)

    var i = 0
    while(i < A.rows) {
      var j = 0
      while(j < A.cols) {
        if (i < j) {
          U(i, j) =  A(i, j)
        } else if (i == j) {
          U(i, i) = A(i, i)
          L(i, i) = 1.0
        } else {
          L(i, j) = A(i, j)
        }
        j = j + 1
      }
      i = i + 1
    }
    (L, U)
  }

  /**
   * Reduces the elements of this RDD in a multi-level tree pattern.
   *
   * @param depth suggested depth of the tree (default: 2)
   * @see [[org.apache.spark.rdd.RDD#reduce]]
   */
  def treeReduce[T: ClassTag](rdd: RDD[T], f: (T, T) => T, depth: Int = 2): T = {
    require(depth >= 1, s"Depth must be greater than or equal to 1 but got $depth.")
    val reducePartition: Iterator[T] => Option[T] = iter => {
      if (iter.hasNext) {
        Some(iter.reduceLeft(f))
      } else {
        None
      }
    }
    val partiallyReduced = rdd.mapPartitions(it => Iterator(reducePartition(it)))
    val op: (Option[T], Option[T]) => Option[T] = (c, x) => {
      if (c.isDefined && x.isDefined) {
        Some(f(c.get, x.get))
      } else if (c.isDefined) {
        c
      } else if (x.isDefined) {
        x
      } else {
        None
      }
    }
    treeAggregate(Option.empty[T])(partiallyReduced, op, op, depth)
      .getOrElse(throw new UnsupportedOperationException("empty collection"))
  }

  /**
   * Aggregates the elements of this RDD in a multi-level tree pattern.
   *
   * @param depth suggested depth of the tree (default: 2)
   * @see [[org.apache.spark.rdd.RDD#aggregate]]
   */
  def treeAggregate[T: ClassTag, U: ClassTag](zeroValue: U)(
      rdd: RDD[T],
      seqOp: (U, T) => U,
      combOp: (U, U) => U,
      depth: Int = 2): U = {
    require(depth >= 1, s"Depth must be greater than or equal to 1 but got $depth.")
    if (rdd.partitions.size == 0) {
      return zeroValue
    }
    val aggregatePartition = (it: Iterator[T]) => it.aggregate(zeroValue)(seqOp, combOp)
    var partiallyAggregated = rdd.mapPartitions(it => Iterator(aggregatePartition(it)))
    var numPartitions = partiallyAggregated.partitions.size
    val scale = math.max(math.ceil(math.pow(numPartitions, 1.0 / depth)).toInt, 2)
    // If creating an extra level doesn't help reduce the wall-clock time, we stop tree aggregation.
    while (numPartitions > 1) { // while (numPartitions > scale + numPartitions / scale) {
      numPartitions /= scale
      val curNumPartitions = numPartitions
      partiallyAggregated = partiallyAggregated.mapPartitionsWithIndex { (i, iter) =>
        iter.map((i % curNumPartitions, _))
      }.reduceByKey(new HashPartitioner(curNumPartitions), combOp).values
    }
    partiallyAggregated.reduce(combOp)
  }

  def aboutEq(a: DenseMatrix[Double], b: DenseMatrix[Double], thresh: Double = 1e-8) = {
    math.abs(max(a-b)) < thresh
  }

  // Creates a coalescer that can be used on RDDs which have same number of partitions
  // and same number of rows per partition.
  // This is useful as many RDDs can be coalesced in a similar fashion.
  def createCoalescer[T: ClassTag](firstRDD: RDD[T], numPartitions: Int) = {
    // assert(rdds.length > 0)
    // // First get a random RDD of indices
    // val firstRDD = rdds(0)
    val distributePartition = (index: Int, items: Iterator[_]) => {
      var position = (new Random(index)).nextInt(numPartitions)
      items.map { t =>
        // Note that the hash code of the key will just be the key itself. The HashPartitioner
        // will mod it with the number of total partitions.
        position = position + 1
        position
      }
    } : Iterator[Int]

    val randomIndices = firstRDD.mapPartitionsWithIndex(distributePartition)
    val partitioner = new HashPartitioner(numPartitions)

    val coalescer = new Coalescer(randomIndices, partitioner)
    coalescer
  }

  class Coalescer(randomIndices: RDD[Int], partitioner: Partitioner) {
    def apply[T: ClassTag](rdd: RDD[T]) = {
      randomIndices.zip(rdd).partitionBy(partitioner).values
    }
  }

  def repartitionAndSortWithinPartitions[K : Ordering : ClassTag, V: ClassTag](rdd: RDD[(K, V)], partitioner: Partitioner): RDD[(K, V)] = {
    val ordering = implicitly[Ordering[K]]
    new ShuffledRDD[K, V, V](rdd, partitioner).setKeyOrdering(ordering)
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
  }

  def calcTestErr(test: RowPartitionedMatrix,
    x: DenseMatrix[Double],
    actualLabels: RDD[Array[Int]], k: Int): Double = {

    // Compute number of test images
    val numTestImages = test.numRows().toInt

    // Broadcast x
    val xBroadcast = test.rdd.context.broadcast(x)

    // Calculate predictions
    val prediction = test.rdd.map { mat =>
      mat.mat * xBroadcast.value
    }

    val predictionArray = prediction.flatMap { p =>
      p.data.grouped(p.rows).toSeq.transpose.map(y => y.toArray)
    }

    val predictedLabels = topKClassifier(k, predictionArray)
    val errPercent = getErrPercent(predictedLabels, actualLabels, numTestImages)
    errPercent
  }

  def calcFusedTestErr(daisyTest: RowPartitionedMatrix, lcsTest: RowPartitionedMatrix,
    daisyX: DenseMatrix[Double], lcsX: DenseMatrix[Double],
    actualLabels: RDD[Array[Int]],
    daisyWt: Double, lcsWt: Double, k: Int): Double = {

      // Compute number of test images
      val numTestImages = daisyTest.numRows().toInt

      // Broadcast x
      val daisyXBroadcast = daisyTest.rdd.context.broadcast(daisyX)
      val lcsXBroadcast = lcsTest.rdd.context.broadcast(lcsX)

      // Calculate predictions
      val daisyPrediction = daisyTest.rdd.map { mat =>
        mat.mat * daisyXBroadcast.value
      }
      val lcsPrediction = lcsTest.rdd.map { mat =>
        mat.mat * lcsXBroadcast.value
      }

      // Fuse b matrices
      val fusedPrediction = daisyPrediction.zip(lcsPrediction).flatMap { p =>
        val fused = (p._1*daisyWt + p._2*lcsWt)
        // Convert from DenseMatrix to Array[Array[Double]]
        fused.data.grouped(fused.rows).toSeq.transpose.map(x => x.toArray)
      }

      val predictedLabels = Utils.topKClassifier(k, fusedPrediction)
      val errPercent = Utils.getErrPercent(predictedLabels, actualLabels, numTestImages)
      errPercent
  }

  def calcFusedTestErrors(daisyTests: Seq[RowPartitionedMatrix], lcsTests: Seq[RowPartitionedMatrix],
    daisyXs: Seq[DenseMatrix[Double]], lcsXs: Seq[DenseMatrix[Double]],
    actualLabels: RDD[Array[Int]],
    daisyWt: Double, lcsWt: Double, k:Int ): DenseVector[Double] = {

      // Compute number of test images
      val l = daisyTests.length
      var i = 0
      val numTestImages = daisyTests(0).numRows().toInt

      var runningSum : Option[RDD[Array[Double]]] = None
      val testErrors = DenseVector.zeros[Double](l)

      while (i < l) {
        val A = daisyTests(i)
        val x = A.rdd.context.broadcast(daisyXs(i))
        val B = lcsTests(i)
        val y = B.rdd.context.broadcast(lcsXs(i))
        val Ax = A.rdd.map( mat => mat.mat*x.value)
        val By = B.rdd.map( mat => mat.mat*y.value)
        val fusedPrediction = Ax.zip(By).flatMap { p =>
          val fused = (p._1*daisyWt + p._2*lcsWt)
          fused.data.grouped(fused.rows).toSeq.transpose.map(x => x.toArray)
        }
        if (runningSum.isEmpty) {
          runningSum = Some(fusedPrediction)
        } else {
          runningSum = Some(runningSum.get.zip(fusedPrediction).map(p => p._1.zip(p._2).map( y =>
            y._1 + y._2)))
          }
          val predictedLabels = topKClassifier(k, runningSum.get)
          val errPercent = getErrPercent(predictedLabels, actualLabels, numTestImages)
          testErrors(i) = errPercent
          i = i + 1
        }
        testErrors
  }

  def calcTestErrors(tests: Seq[RowPartitionedMatrix],
    xs: Seq[DenseMatrix[Double]],
    actualLabels: RDD[Array[Int]], k:Int): DenseVector[Double] = {

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
          val predictedLabels = topKClassifier(k, runningSum.get)
          val errPercent = getErrPercent(predictedLabels, actualLabels, numTestImages)
          testErrors(i) = errPercent
          i = i + 1
        }
        testErrors
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

  def loadMatrixFromFile(sc: SparkContext, filename: String, parts: Int): RDD[Array[Double]] = {
    sc.textFile(filename, parts).map { line =>
      line.split(",").map(y => y.toDouble)
    }
  }

}
