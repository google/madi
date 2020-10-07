# Lint as: python3
#     Copyright 2020 Google LLC
#
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.
"""Utility for variable attribution using Integrated Gradients."""
from typing import Tuple, Dict, Optional

from absl import logging
from madi.detectors.base_interpreter import BaseAnomalyInterpreter
import numpy as np
import pandas as pd
from scipy.spatial import distance
import tensorflow as tf

_CLASS_PROB_LABEL = 'class_prob'


class IntegratedGradientsInterpreter(BaseAnomalyInterpreter):
  """Utility for variable attribution using Integrated Gradients.

  Solution is based on Axiomatic Attribution for Deep Networks (2017),
  Sundararajan, M., Tuly, A., Yan, Q.
  https://drive.google.com/file/d/0B0ms91L2bD9GaFNaS19Ha1UxNzQ/view

  Given a reference point and a sample point of length N, we wish to determine
  the distance as the integral of gradients, from sample to reference.

  To compute the integral numerically, we compute and sum the integrals over an
  fixed number of steps between the two points for each variable. This solution
  is based on using the Keras optimization.get_gradients() method.

  The NN model trained to detect anomalies is queried for variable attribution.

  In this application, the baseline is the closest conformal point. The
  IG results which variable(s) caused the model to claim the point to be
  anomalous.
  """

  def __init__(self,
               model: tf.keras.Sequential,
               df_pos_normalized: pd.DataFrame,
               min_baseline_class_conf: float,
               baseline_size_limit: int,
               num_steps_integrated_gradients: int = 2000):
    """Constructs an interpreter based on integrated gradients.

    Args:
      model: tf.keras.Sequential model from NS-NN Anomaly Detector.
      df_pos_normalized: Dataframe with positive sample (normalized).
      min_baseline_class_conf: minimum classification score to be in baseline
      baseline_size_limit: choose the top confr that meet the min conf
      num_steps_integrated_gradients: number steps in integrated gradients.

    Raises:
      NoQualifyingBaselineError: if there are no points in the baseline, which
      provides the maximum confidence score.
    """

    # To compute integrated gradients we need a model.
    self._model = model
    self._num_steps_integrated_gradients = num_steps_integrated_gradients
    self._df_baseline, max_class_confidence = select_baseline(
        df_pos_normalized=df_pos_normalized,
        model=model,
        min_p=min_baseline_class_conf,
        max_count=baseline_size_limit)
    if self._df_baseline.empty:
      raise NoQualifyingBaselineError(min_baseline_class_conf,
                                      max_class_confidence)

    # Create a gradient tensor, where the loss tensor is the output, and the
    # list of variables (params) are the input.
    logging.info('Finished setting up IG.')

  def explain(self,
              sample: np.ndarray,
              reference: np.ndarray,
              num_steps: int = 50) -> (Tuple[np.ndarray, np.matrix]):
    """Returns variable attribution between sample and reference.

    Args:
      sample: normalized observed vector length N
      reference: normalized reference vector length N, as described in IG paper
      num_steps: number of steps to approximate the path integral between sample
        and reference.  Returns a normalized vector length N, that indicates
        magnitude of the variables in [-1, 1].
    """

    # Create the intermediate steps between sample and reference.
    interp_output = tf.convert_to_tensor(
        np.linspace(reference, sample, num_steps))
    with tf.GradientTape() as tape:
      tape.watch(interp_output)
      outs = self._model(interp_output)

    # Next compute and sum the gradients.
    gradients = tape.gradient(outs, interp_output)
    mat_grad = np.matrix(gradients)
    dif = np.array(reference - sample)

    # Integrated Gradients dif * grad_sum for each variable i:
    # ig_i = (x_i - x'_i) * integral over alpha from 0 to 1 on dF/dx_i(alpha)
    # where sample = x_i, reference  = x'_i, and alpha in [0, 1]
    grad_sum = np.array(mat_grad.sum(axis=0).tolist()[0])

    attribution = np.multiply(grad_sum, dif) / float(num_steps)
    denom = sum([abs(val) for val in attribution])
    logging.info('Sum of attributions %f', abs(sum(attribution)))

    # We've found it useful to noirmalize the the total attribution, A to
    # distribute the blame fully on the dimensions. In cases where the
    # the total attrbution by IG will sum to the difference between the
    # baseline and the observed point, which could be less than one. In order
    # to sum the blame to one across all dimensions, we divide by the sum.
    if denom == 0:
      return np.zeros(len(attribution)), mat_grad
    return attribution / denom, mat_grad

  def blame(
      self,
      observation_normalized: pd.Series,
  ) -> (Tuple[Dict[str, float], Dict[str, float], Optional[pd.DataFrame]]):
    """Performs variable attribution using a baseline and integrated grads.

    Args:
      observation_normalized: original feature names as cols and values.

    Returns:
      Attribution Dict with variable name key, and proportional blame as value
      Reference Dict: Nearest original baseline point with variable name as kay
      Gradiant Matrix useful to disply and understand the gradiants.
    """

    attribution_dict = {}
    reference_point_dict = {}

    observation_normalized_array = observation_normalized.to_numpy()
    # Pull out the point in the baseline sample.
    nearest_reference_index, _ = find_nearest_euclidean(
        self._df_baseline, observation_normalized_array)

    # Use the normalized reference point for integrated gradients.
    reference_point_normalized = self._df_baseline.loc[[
        nearest_reference_index
    ]]

    # Use integrated gradients to compute variable attribution info,
    # including reference point and integrated gradients attribution.
    explanation, mat_grad = self.explain(
        observation_normalized_array,
        reference_point_normalized.iloc[0].to_numpy(),
        num_steps=self._num_steps_integrated_gradients)

    df_grad = pd.DataFrame(mat_grad)

    # Attribution is the proportional explanation.
    attribution = [abs(v) for v in explanation]
    for i, column in enumerate(observation_normalized.index):
      attribution_dict[column] = attribution[i]
      reference_point_dict[column] = reference_point_normalized.iloc[0][column]

    return attribution_dict, reference_point_dict, df_grad


def find_nearest_euclidean(df_reference: pd.DataFrame,
                           example: np.ndarray) -> (Tuple[int, float]):
  """Returns index, dist from df_reference closest to example.

  Based on Section 3.2 (Interpreting Anomalies with Integrated Gradients)
  Interpretable, Multidimensional, Multimodal Anomaly Detection with Negative
  Sampling (Sipple 2020).

  Args:
    df_reference: baseline data set for anomaly detection.
    example: feature values of anomalous datapoint

  Returns:
    tuple with dataframe index of the neearest point and its Euclidian distance

  """
  for column in df_reference:
    if not np.issubdtype(df_reference[column].dtype, np.number):
      raise ValueError('The feature column %s is not numeric.' % column)
  if len(example) != df_reference.shape[1]:
    raise ValueError('Dimensionality is not the same: %d != %d' %
                     (df_reference.shape[1], len(example)))

  # Minor fix required to handle a pandas bug, adding v = example
  # Pandas issue: https://github.com/pandas-dev/pandas/issues/36948
  # and PR: https://github.com/pandas-dev/pandas/pull/36950
  distances = df_reference.agg(distance.euclidean, 1, v=example)
  nearest_ix = distances.idxmin()
  return nearest_ix, distances[nearest_ix]


def select_baseline(df_pos_normalized: pd.DataFrame,
                    model: tf.keras.Sequential,
                    min_p: float = 0.85,
                    max_count: int = 100):
  """Selects the representative subsamnple that will be used as baselines.

  Based on Proposition 3 (Baseline Set for Anomaly Detection) of
  Interpretable, Multidimensional, Multimodal Anomaly Detection with Negative
  Sampling (Sipple 2020).

  Args:
    df_pos_normalized: data frame of the normalized positive sample.
    model: classifier model from NS-NN.
    min_p: minimum class score to be be considered as a baseline normal.
    max_count: maximum number of reference points to be selected.

  Returns:
    data frame of the normalized baseline and the maximum conf score
  """
  x = np.float32(np.matrix(df_pos_normalized))
  y_hat = model.predict(x, verbose=1, steps=1)
  df_pos_normalized[_CLASS_PROB_LABEL] = y_hat
  high_scoring_predictions = df_pos_normalized[
      df_pos_normalized[_CLASS_PROB_LABEL] >= min_p]
  high_scoring_predictions = high_scoring_predictions.sort_values(
      by=_CLASS_PROB_LABEL, ascending=False)
  high_scoring_predictions = high_scoring_predictions.drop(
      columns=[_CLASS_PROB_LABEL])
  return high_scoring_predictions[:max_count], float(max(y_hat))


class Error(Exception):
  """Base class for exceptions in integrated gradiants interpreter."""
  pass


class NoQualifyingBaselineError(BaseException):
  """Exception rasied when there are no baseline points."""

  def __init__(self, min_class_confidence: float,
               highest_class_confidence: float):
    """Constructs an exception when there are no baseline points.

    Args:
      min_class_confidence: threshold set when constructing the interpreter
      highest_class_confidence: highest score the model produced on sample
    """
    super(NoQualifyingBaselineError, self).__init__(min_class_confidence,
                                                    highest_class_confidence)
    self.highest_class_confidence = highest_class_confidence
    self.min_class_confidence = min_class_confidence
    self.message = ('No positive sample points met the min_class_confidence '
                    '%3.2f. Highest class confidence is %3.2f') % (
                        min_class_confidence, highest_class_confidence)
