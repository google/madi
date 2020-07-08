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
import numpy as np
import pandas as pd
import sklearn.ensemble


class IsolationForestAd(sklearn.ensemble.IsolationForest,
                        BaseAnomalyDetectionAlgorithm):
  """Wrapper class around the scikit-learn Isolation Forest Implementation."""

  def train_model(self, x_train: pd.DataFrame):
    super(IsolationForestAd, self).fit(X=x_train)

  def predict(self, x_test: pd.DataFrame) -> pd.DataFrame:
    preds = super(IsolationForestAd, self).predict(x_test)
    x_test['class_prob'] = np.where(preds == -1, 0, preds)
    return x_test
