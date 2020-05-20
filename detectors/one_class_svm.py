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
import numpy as np
import pandas as pd
import sklearn.svm


class OneClassSVMAd(sklearn.svm.OneClassSVM, BaseAnomalyDetectionAlgorithm):
  """Wrapper class around the scikit-learn OC-SVM implementation."""

  def train_model(self, x_train: pd.DataFrame) -> None:
    """Trains a OC-SVM Anomaly detector using the positive sample.

    Args:
      x_train: training sample, with numeric feature columns.
    """
    super(OneClassSVMAd, self).fit(X=x_train)

  def predict(self, x_test: pd.DataFrame) -> pd.DateOffset:
    """Performs anomaly detection on a new sample.

    Args:
      x_test: dataframe with the new datapoints.

    Returns:
      original dataframe with a new column labled 'class_prob' as 1.0
      for normal to 0.0 for anomalous.
    """
    preds = super(OneClassSVMAd, self).predict(x_test)
    x_test['class_prob'] = np.where(preds == -1, 0, preds)
    return x_test
