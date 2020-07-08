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
"""Base class for anomaly detection datasets."""

import abc
import os
import typing

from madi.utils import file_utils
import pandas as pd


class BaseDataset(metaclass=abc.ABCMeta):
  """All AD algorithms will include a train and predict step."""

  @property
  @abc.abstractmethod
  def sample(self) -> pd.DataFrame:
    """Trains the model on a training set."""
    pass

  @property
  @abc.abstractmethod
  def name(self) -> str:
    """Unique name of the data set."""
    pass

  @property
  @abc.abstractmethod
  def description(self) -> str:
    """Informative text summary of the dataset."""
    pass

  def _load_readme(
      self, readmefile: typing.Union[str, os.PathLike,
                                     file_utils.PackageResource]
  ) -> str:
    with file_utils.open_text_resource(readmefile) as text_file:
      return " ".join(text_file)
