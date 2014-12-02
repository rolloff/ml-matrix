package edu.berkeley.cs.amplab.mlmatrix

import breeze.linalg._

import org.apache.spark.rdd.RDD
import org.apache.spark.storage.StorageLevel

/**
 * Regression model trained using LinearRegression.
 *
 * @param weights Weights computed for every feature.
 * @param intercept Intercept computed for this model.
 */
class MultiClassLinearRegressionModel (
    val weights: DenseMatrix[Double]) extends Serializable {

  def predict(data: DenseVector[Double]): DenseVector[Double] = {
    weights * data
  }
}

/**
 * Train a linear regression model with no regularization using Stochastic Gradient Descent.
 * This solves the least squares regression formulation
 *              f(weights) = 1/n ||A weights-y||^2
 * (which is the mean squared error).
 * Here the data matrix has n rows, and the input RDD holds the set of rows of A, each with
 * its corresponding right hand side label y.
 * See also the documentation for the precise formulation.
 */
class MultiClassLinearRegressionWithSGD (
    private var stepSize: Double,
    private var numIterations: Int,
    private var miniBatchFraction: Double,
    private var regParam: Double) extends Serializable with Logging {

  private val gradient = new LeastSquaresGradient()
  private val updater = new SquaredL2Updater()
  val optimizer = new GradientDescent(gradient, updater)
    .setStepSize(stepSize)
    .setNumIterations(numIterations)
    .setMiniBatchFraction(miniBatchFraction)
    .setRegParam(regParam)

  /**
   * Construct a LinearRegression object with default parameters: {stepSize: 1.0,
   * numIterations: 100, miniBatchFraction: 1.0}.
   */
  def this() = this(1.0, 100, 1.0, 0.0)


  /**
   * Run the algorithm with the configured parameters on an input
   * RDD of LabeledPoint entries.
   */
  def run(input: RDD[(Array[Double], DenseVector[Double])]): MultiClassLinearRegressionModel = {
    val first = input.first()
    val numFeatures: Int = first._2.size
    val numClasses: Int = first._1.length

    val initialWeights = DenseMatrix.zeros[Double](numFeatures, numClasses)
    // val data = input.map(labeledPoint => (labeledPoint.label, labeledPoint.features))
    val data = input
    val weights = optimizer.optimize(data, initialWeights)

    // Warn at the end of the run as well, for increased visibility.
    if (input.getStorageLevel == StorageLevel.NONE) {
      logWarning("The input data was not directly cached, which may hurt performance if its"
        + " parent RDDs are also uncached.")
    }

    new MultiClassLinearRegressionModel(weights)
  }
}
