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
"""Wrapper for One-Class SVM Anomaly Detector based on sklearn."""
from madi.detectors.base_detector import BaseAnomalyDetectionAlgorithm
import madi.utils.sample_utils as sample_utils
import numpy as np
import pandas as pd
import sklearn.svm


class OneClassSVMAd(sklearn.svm.OneClassSVM, BaseAnomalyDetectionAlgorithm):
  """Wrapper class around the scikit-learn OC-SVM implementation."""

  def __init__(self, *args, **kwargs):
    """Constructs a OC-SVM Anomaly Detector.

    Args:
      *args: See the klearn.svm.OneClassSVM.
      **kwargs: See the klearn.svm.OneClassSVM.
    """
    super(OneClassSVMAd, self).__init__(*args, **kwargs)
    self._normalization_info = None

  def train_model(self, x_train: pd.DataFrame) -> None:
    """Trains a OC-SVM Anomaly detector using the positive sample.

    Args:
      x_train: training sample, with numeric feature columns.
    """
    self._normalization_info = sample_utils.get_normalization_info(x_train)
    column_order = sample_utils.get_column_order(self._normalization_info)
    normalized_x_train = sample_utils.normalize(x_train[column_order],
                                                self._normalization_info)
    super(OneClassSVMAd, self).fit(X=normalized_x_train)

  def predict(self, sample_df: pd.DataFrame) -> pd.DateOffset:
    """Performs anomaly detection on a new sample.

    Args:
      sample_df: dataframe with the new datapoints.

    Returns:
      original dataframe with a new column labled 'class_prob' as 1.0
      for normal to 0.0 for anomalous.
    """
    sample_df_normalized = sample_utils.normalize(sample_df,
                                                  self._normalization_info)
    column_order = sample_utils.get_column_order(self._normalization_info)
    x_test = np.float32(np.matrix(sample_df_normalized[column_order]))
    preds = super(OneClassSVMAd, self).predict(x_test)
    sample_df['class_prob'] = np.where(preds == -1, 0, preds)
    return sample_df
