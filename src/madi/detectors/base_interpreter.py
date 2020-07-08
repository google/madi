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
"""Base class for Interpreting Anomalies."""

import abc
from typing import Tuple, Dict, Optional

import pandas as pd


class BaseAnomalyInterpreter(metaclass=abc.ABCMeta):
  """Defines methods for interpreting anomalies."""

  @abc.abstractmethod
  def blame(
      self,
      sample_row: pd.Series
  ) -> (Tuple[Dict[str, float], Dict[str, float], Optional[pd.DataFrame]]):
    """Performs variable attribution.

    Args:
      sample_row: a pd.Series with feature names as cols and values.

    Returns:
      Attribution Dict with variable name key, and proportional blame as value
      Reference Dict: Nearest baseline point with variable name as kay
      Gradiant Matrix useful to disply and understand the gradiants.
    """
    pass
