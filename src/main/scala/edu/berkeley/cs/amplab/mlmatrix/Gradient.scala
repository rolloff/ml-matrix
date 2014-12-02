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

/**
 * Class used to compute the gradient for a loss function, given a single data point.
 */
abstract class Gradient extends Serializable {
  /**
   * Compute the gradient and loss given the features of a single data point,
   * add the gradient to a provided vector to avoid creating new objects, and return loss.
   *
   * @param data features for one data point
   * @param label label for this data point
   * @param weights weights/coefficients corresponding to features
   * @param cumGradient the computed gradient will be added to this vector
   *
   * @return loss
   */
  def compute(data: DenseVector[Double], label: Array[Double], weights: DenseMatrix[Double], cumGradient: DenseMatrix[Double]): Double
}


/**
 * Compute gradient and loss for a Least-squared loss function, as used in linear regression.
 * This is correct for the averaged least squares loss function (mean squared error)
 *              L = 1/n ||A weights-y||^2
 * See also the documentation for the precise formulation.
 */
class LeastSquaresGradient extends Gradient {

  override def compute(
      data: DenseVector[Double],
      labels: Array[Double],
      weights: DenseMatrix[Double],
      cumGradient: DenseMatrix[Double]): Double = {

    val breezeLabels = new DenseVector[Double](labels)
    val dataMat = new DenseMatrix[Double](1, data.length, data.toArray)

    val diff = weights.t * data - breezeLabels

    cumGradient :+=  (kron(dataMat.t, new DenseMatrix[Double](1, diff.length, diff.toArray)) * 2.0)

    math.pow(norm(diff, 2.0), 2)
  }
}

