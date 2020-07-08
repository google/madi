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
"""Isolation Forest Anomaly Detector."""
from madi.detectors.base_detector import BaseAnomalyDetectionAlgorithm
import madi.utils.sample_utils as sample_utils
import numpy as np
import pandas as pd
import sklearn.ensemble

_CLASS_LABEL = 'class_label'
_NORMAL_CLASS = 1


class NegativeSamplingRandomForestAd(sklearn.ensemble.RandomForestClassifier,
                                     BaseAnomalyDetectionAlgorithm):
  """Anomaly Detector with a Random Forest Classifier and negative sampling."""

  def __init__(self, *args, sample_ratio=2.0, sample_delta=0.05, **kwargs):
    """Constructs a NS-RF Anomaly Detector.

    Args:
      *args: See the sklearn.ensemble.RandomForestClassifier.
      sample_ratio: ratio of negative sample size to positive sample size.
      sample_delta: sample extension beypnd min and max limits of pos sample.
      **kwargs: See the sklearn.ensemble.RandomForestClassifier.
    """
    super(NegativeSamplingRandomForestAd, self).__init__(*args, **kwargs)
    self._normalization_info = None
    self._sample_ratio = sample_ratio
    self._sample_delta = sample_delta

  def train_model(self, x_train: pd.DataFrame) -> None:
    """Trains a NS-NN Anomaly detector using the positive sample.

    Args:
      x_train: training sample, which does not need to be normalized.
    """
    # TODO(sipple) Consolidate the normalization code into the base class.
    self._normalization_info = sample_utils.get_normalization_info(x_train)
    column_order = sample_utils.get_column_order(self._normalization_info)
    normalized_x_train = sample_utils.normalize(x_train[column_order],
                                                self._normalization_info)

    normalized_training_sample = sample_utils.apply_negative_sample(
        positive_sample=normalized_x_train,
        sample_ratio=self._sample_ratio,
        sample_delta=self._sample_delta)

    super(NegativeSamplingRandomForestAd, self).fit(
        X=normalized_training_sample[column_order],
        y=normalized_training_sample[_CLASS_LABEL])

  def predict(self, sample_df: pd.DataFrame) -> pd.DataFrame:
    """Performs anomaly detection on a new sample.

    Args:
      sample_df: dataframe with the new datapoints, not normalized.

    Returns:
      original dataframe with a new column labled 'class_prob' rangin from 1.0
      as normal to 0.0 as anomalous.
    """

    sample_df_normalized = sample_utils.normalize(sample_df,
                                                  self._normalization_info)
    column_order = sample_utils.get_column_order(self._normalization_info)
    x = np.float32(np.matrix(sample_df_normalized[column_order]))

    preds = super(NegativeSamplingRandomForestAd, self).predict_proba(x)
    sample_df['class_prob'] = preds[:, _NORMAL_CLASS]
    return sample_df
