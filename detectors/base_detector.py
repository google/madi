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
"""Base class for Anomaly Detector instances."""

import abc

import pandas as pd


class BaseAnomalyDetectionAlgorithm(metaclass=abc.ABCMeta):
  """All AD algorithms will include a train and predict step."""

  @abc.abstractmethod
  def train_model(self, x_train: pd.DataFrame) -> None:
    """Trains the model on a training set."""
    pass

  @abc.abstractmethod
  def predict(self, sample_df: pd.DataFrame) -> pd.DataFrame:
    """Predicts with the model on a test set."""
    pass
