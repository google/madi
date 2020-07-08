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
"""Tests for google3.third_party.py.madi.datasets.forestcover_dataset."""

import os

from absl import logging
from madi.datasets import forestcover_dataset
import tensorflow as tf

_DATA_FILE_IN = os.path.join(
    os.path.dirname(__file__), 'test_data/covtype.test.data')
_DATA_FILE_TEST = 'covtype.data'
_COL_NAMES_SELECT = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'class_label'
]


class TestForestCoverDataset:

  def test_forsetcover_dataset(self, tmpdir):
    datadir = tmpdir
    datafile_test = os.path.join(datadir, _DATA_FILE_TEST)
    logging.info(_DATA_FILE_IN)

    tf.io.gfile.copy(_DATA_FILE_IN, datafile_test, overwrite=True)

    ds = forestcover_dataset.ForestCoverDataset(datadir)

    assert len(ds.sample) == 139
    assert sorted(ds.sample.columns) == sorted(_COL_NAMES_SELECT)
    assert set(ds.sample.columns) == set(_COL_NAMES_SELECT)
