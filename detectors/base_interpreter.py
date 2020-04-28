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

import numpy as np
import six


@six.add_metaclass(abc.ABCMeta)
class BaseAnomalyInterpreter(object):
  """Defines methods for interpreting anomalies."""

  @abc.abstractmethod
  def blame(self, anomaly: np.array) -> np.array:
    """Accepts a an array of values and returns a proportional blame."""
    pass
