/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package edu.berkeley.cs.amplab.mlmatrix

import breeze.linalg._
import scala.collection.mutable.ArrayBuffer

import edu.berkeley.cs.amplab.mlmatrix.util.Utils

import org.apache.spark.rdd.RDD

/**
 * Class used to solve an optimization problem using Gradient Descent.
 * @param gradient Gradient function to be used.
 * @param updater Updater to be used to update weights after every iteration.
 */
class GradientDescent (private var gradient: Gradient, private var updater: Updater) extends Logging {

  private var stepSize: Double = 1.0
  private var numIterations: Int = 100
  private var regParam: Double = 0.0
  private var miniBatchFraction: Double = 1.0

  /**
   * Set the initial step size of SGD for the first step. Default 1.0.
   * In subsequent steps, the step size will decrease with stepSize/sqrt(t)
   */
  def setStepSize(step: Double): this.type = {
    this.stepSize = step
    this
  }

  /**
   * :: Experimental ::
   * Set fraction of data to be used for each SGD iteration.
   * Default 1.0 (corresponding to deterministic/classical gradient descent)
   */
  def setMiniBatchFraction(fraction: Double): this.type = {
    this.miniBatchFraction = fraction
    this
  }

  /**
   * Set the number of iterations for SGD. Default 100.
   */
  def setNumIterations(iters: Int): this.type = {
    this.numIterations = iters
    this
  }

  /**
   * Set the regularization parameter. Default 0.0.
   */
  def setRegParam(regParam: Double): this.type = {
    this.regParam = regParam
    this
  }

  /**
   * Set the gradient function (of the loss function of one single data example)
   * to be used for SGD.
   */
  def setGradient(gradient: Gradient): this.type = {
    this.gradient = gradient
    this
  }


  /**
   * Set the updater function to actually perform a gradient step in a given direction.
   * The updater is responsible to perform the update from the regularization term as well,
   * and therefore determines what kind or regularization is used, if any.
   */
  def setUpdater(updater: Updater): this.type = {
    this.updater = updater
    this
  }

  /**
   * Runs gradient descent on the given training data.
   * @param data training data
   * @param initialWeights initial weights
   * @return solution vector
   */
  def optimize(data: RDD[(Array[Double], DenseVector[Double])],
    initialWeights: DenseMatrix[Double]): DenseMatrix[Double] = {
    val (weights, _) = GradientDescent.runMiniBatchSGD(
      data,
      gradient,
      updater,
      stepSize,
      numIterations,
      regParam,
      miniBatchFraction,
      initialWeights)
    weights
  }

}

/**
 * Top-level method to run gradient descent.
 */
object GradientDescent extends Logging {

  def aggregateParts(part: Iterator[(Array[Double], DenseVector[Double])]) 
      : Iterator[(DenseMatrix[Double], DenseMatrix[Double])] = {
    val partSeq = part.toSeq
    val numRows = partSeq.length
    val numCols = partSeq(0)._2.length
    val numClasses = partSeq(0)._1.length

    val features = new Array[Double](numRows * numCols)
    val labels = new Array[Double](numRows * numClasses)
    // var numRows = 0
    // var numCols = 0
    // var numClasses = 0
    var i = 0
    var featuresPtr = 0
    var labelsPtr = 0
    while (i < numRows) {
      val nextItem = partSeq(i)
      System.arraycopy(nextItem._2.data, 0, features, featuresPtr, nextItem._2.length)
      featuresPtr += nextItem._2.length
      // features ++= nextItem._2.data
      System.arraycopy(nextItem._1, 0, labels, labelsPtr, nextItem._1.length)
      labelsPtr += nextItem._1.length

      i += 1

      // labels ++= nextItem._1
      // numRows += 1
      // numCols = nextItem._2.length
      // numClasses = nextItem._1.length
    }
    val featuresMat = new DenseMatrix(numCols, numRows, features.toArray)
    val labelsMat = new DenseMatrix(numClasses, numRows, labels.toArray)
    Iterator.single((labelsMat.t, featuresMat.t))
  }

  /**
   * Run stochastic gradient descent (SGD) in parallel using mini batches.
   * In each iteration, we sample a subset (fraction miniBatchFraction) of the total data
   * in order to compute a gradient estimate.
   * Sampling, and averaging the subgradients over this subset is performed using one standard
   * spark map-reduce in each iteration.
   *
   * @param data - Input data for SGD. RDD of the set of data examples, each of
   *               the form (label, [feature values]).
   * @param gradient - Gradient object (used to compute the gradient of the loss function of
   *                   one single data example)
   * @param updater - Updater function to actually perform a gradient step in a given direction.
   * @param stepSize - initial step size for the first step
   * @param numIterations - number of iterations that SGD should be run.
   * @param regParam - regularization parameter
   * @param miniBatchFraction - fraction of the input data set that should be used for
   *                            one iteration of SGD. Default value 1.0.
   *
   * @return A tuple containing two elements. The first element is a column matrix containing
   *         weights for every feature, and the second element is an array containing the
   *         stochastic loss computed for every iteration.
   */
  def runMiniBatchSGD(
      data: RDD[(Array[Double], DenseVector[Double])],
      gradient: Gradient,
      updater: Updater,
      stepSize: Double,
      numIterations: Int,
      regParam: Double,
      miniBatchFraction: Double,
      initialWeights: DenseMatrix[Double]): (DenseMatrix[Double], Array[Double]) = {

    val stochasticLossHistory = new ArrayBuffer[Double](numIterations)

    val numExamples = data.count()

    // if no data, return initial weights to avoid NaNs
    if (numExamples == 0) {
      logWarning("GradientDescent.runMiniBatchSGD returning initial weights, no data found")
      return (initialWeights, stochasticLossHistory.toArray)
    }

    if (numExamples * miniBatchFraction < 1) {
      logWarning("The miniBatchFraction is too small")
    }

    // Initialize weights as a column vector
    var weights = initialWeights
    val n = weights.size

    /**
     * For the first iteration, the regVal will be initialized as sum of weight squares
     * if it's L2 updater; for L1 updater, the same logic is followed.
     */
    var regVal = updater.compute(
      weights, new DenseMatrix(weights.rows, weights.cols), 0, 1, regParam)._2

    for (i <- 1 to numIterations) {
      val bcWeights = data.context.broadcast(weights)
      // Sample a subset (fraction miniBatchFraction) of the total data
      // compute and sum up the subgradients on this subset (this is one map-reduce)
      val sampleData =  data.sample(false, miniBatchFraction, 123 + i).mapPartitions(aggregateParts)
      // sampleData.cache().count

      val depth = math.ceil(math.log(data.partitions.size)/math.log(2)).toInt

      val (gradientSum, lossSum, miniBatchSize) = 
        Utils.treeAggregate((DenseMatrix.zeros[Double](weights.rows, weights.cols), 0.0, 0L))(
          rdd = sampleData,
          depth = depth,
          seqOp = (c, v: (DenseMatrix[Double], DenseMatrix[Double])) => {
            // c: (grad, loss, count), v: (label, features)
            val l = gradient.compute(v._2, v._1, bcWeights.value, c._1)
            (c._1, c._2 + l, c._3 + v._2.rows)
          },
          combOp = (c1, c2) => {
            // c: (grad, loss, count)
            (c1._1 += c2._1, c1._2 + c2._2, c1._3 + c2._3)
        })

      // sampleData.unpersist()

      if (miniBatchSize > 0) {
        logInfo("At iteration " + i + " stochastic loss " + lossSum + " " + miniBatchSize + " " + lossSum / miniBatchSize
          + " regVal " + regVal)
        /**
         * NOTE(Xinghao): lossSum is computed using the weights from the previous iteration
         * and regVal is the regularization value computed in the previous iteration as well.
         */
        stochasticLossHistory.append(lossSum / miniBatchSize + regVal)

        val update = updater.compute(
          weights, gradientSum :/ miniBatchSize.toDouble, stepSize, i, regParam)
        weights = update._1
        regVal = update._2
      } else {
        logWarning(s"Iteration ($i/$numIterations). The size of sampled batch is zero")
      }
    }

    logInfo("GradientDescent.runMiniBatchSGD finished. Last 20 stochastic losses %s".format(
      stochasticLossHistory.takeRight(20).mkString(", ")))
    (weights, stochasticLossHistory.toArray)
  }
}
