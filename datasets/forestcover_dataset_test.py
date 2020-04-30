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
from absl.testing import absltest
from madi.datasets import forestcover_dataset
import tensorflow as tf

_DATA_FILE_IN = 'data/covtype.test.data'
_DATA_FILE_TEST = 'covtype.data'
_COL_NAMES_SELECT = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'class_label'
]


class ForestCoverDatasetTest(absltest.TestCase):

  def test_forsetcover_dataset(self):
    datadir = self.create_tempdir()
    datafile_in = os.path.join(os.path.dirname(__file__), _DATA_FILE_IN)
    datafile_test = os.path.join(datadir, _DATA_FILE_TEST)
    logging.info(datafile_in)

    tf.io.gfile.copy(datafile_in, datafile_test, overwrite=True)

    ds = forestcover_dataset.ForestCoverDataset(datadir)
    self.assertCountEqual(ds.sample.columns, _COL_NAMES_SELECT)
    self.assertLen(ds.sample, 139)
    logging.info(ds.sample)


if __name__ == '__main__':
  absltest.main()
