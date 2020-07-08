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
import os
import typing

from madi.datasets.base_dataset import BaseDataset
from madi.utils import file_utils
import numpy as np
import pandas as pd

_RESOURCE_LOCATION = "madi.datasets.data"
_DATA_FILE = file_utils.PackageResource(
    _RESOURCE_LOCATION, "anomaly_detection_sample_1577622599.csv")
_README_FILE = file_utils.PackageResource(
    _RESOURCE_LOCATION, "anomaly_detection_sample_1577622599_README.md")

_FileType = typing.Union[str, os.PathLike, file_utils.PackageResource]


class SmartBuildingsDataset(BaseDataset):
  """Smart Buildings data set for Multivariate Anomaly Detection."""

  def __init__(self,
               datafilepath: _FileType = _DATA_FILE,
               readmefilepath: _FileType = _README_FILE):
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

  def _load_data_file(self, datafile: _FileType) -> pd.DataFrame:
    with file_utils.open_text_resource(datafile) as csv_file:
      sample = pd.read_csv(csv_file, header="infer", index_col=0)

    return sample.reindex(np.random.permutation(sample.index))
