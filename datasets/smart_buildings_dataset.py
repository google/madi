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
"""Provides access to the Smart Buidlings dataset for Anomaly Detection."""
from madi.datasets.base_dataset import BaseDataset
import pandas as pd
import tensorflow as tf

_DATA_FILE = "third_party/py/madi/datasets/data/anomaly_detection_sample_1577622599.csv"
_README_FILE = "third_party/py/madi/datasets/data/anomaly_detection_sample_1577622599_README.md"


class SmartBuildingsDataset(BaseDataset):
  """Smart Buildings data set for Multivariate Anomaly Detection."""

  def __init__(self,
               datafilepath: str = _DATA_FILE,
               readmefilepath: str = _README_FILE):
    self._sample = self._load_data_file(datafilepath)
    self._description = self._load_readme(readmefilepath)

  @property
  def sample(self) -> pd.DataFrame:
    return self._sample

  @property
  def name(self) -> str:
    return "smart_buildings"

  @property
  def description(self) -> str:
    return self._description

  def _load_data_file(self, datafile: str) -> pd.DataFrame:
    sample = None
    if not tf.io.gfile.exists(datafile):
      raise AssertionError("{} does not exist".format(datafile))
    with tf.io.gfile.GFile(datafile) as csv_file:
      sample = pd.read_csv(csv_file, header="infer", index_col=0)
    return sample
