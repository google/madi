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
"""Provides access to the UCI Forest Cover dataset for Anomaly Detection."""
import os

from absl import logging
from madi.datasets.base_dataset import BaseDataset
from madi.utils import file_utils
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets.public_api as tfd

_DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz'
_DATA_FILE = 'covtype.data'
_DATA_NAME = 'forest_cover'

_COL_NAMES_SELECT = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'Cover_Type'
]

_COL_NAMES_ALL = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1',
    'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type1',
    'Soil_Type2', 'Soil_Type3', 'Soil_Type4', 'Soil_Type5', 'Soil_Type6',
    'Soil_Type7', 'Soil_Type8', 'Soil_Type9', 'Soil_Type10', 'Soil_Type11',
    'Soil_Type12', 'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
    'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
    'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25', 'Soil_Type26',
    'Soil_Type27', 'Soil_Type28', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31',
    'Soil_Type32', 'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
    'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40', 'Cover_Type'
]

_README_FILE = file_utils.PackageResource('madi.datasets.data',
                                          'forestcover_README.md')
_CHECKSUM_DIR = 'google3/madi/datasets/checksum'


class ForestCoverDataset(BaseDataset):
  """Forest Cover dataset for Multivariate Anomaly Detection."""

  def __init__(self, data_dir):
    """Loads and shuffles the Forest Cover datase.

    If the data_dir does not contain the datafile, it will try to download
    the data from the URL directly.

    Args:
      data_dir: directory for the data.
    """

    datafile_in = os.path.join(data_dir, _DATA_FILE)
    logging.info('datafile: %s', datafile_in)

    if not tf.io.gfile.exists(datafile_in):

      logging.info('adding %s as checksum dir', _CHECKSUM_DIR)
      tfd.core.download.checksums.add_checksums_dir(_CHECKSUM_DIR)
      dm = tfd.core.download.download_manager.DownloadManager(
          download_dir=data_dir,
          register_checksums=True,
          dataset_name=_DATA_NAME)
      raw_extracted = dm.download_and_extract(_DATASET_URL)
      tf.io.gfile.copy(raw_extracted, datafile_in, overwrite=True)

    if not tf.io.gfile.exists(datafile_in):
      raise AssertionError('{} does not exist'.format(_DATA_FILE))

    with tf.io.gfile.GFile(datafile_in) as csv_file:
      input_df = pd.read_csv(
          csv_file, names=_COL_NAMES_ALL, usecols=_COL_NAMES_SELECT)

    input_df = input_df[input_df['Cover_Type'].isin([2, 4])]
    # Randomize the ordering.
    input_df = input_df.reindex(np.random.permutation(input_df.index))
    # Label all points with Cover Type 2 as normal, all with Cover Type 4 as
    # anomalous.
    input_df['class_label'] = [int(t) for t in input_df['Cover_Type'] == 2]
    # Now we can drop the Cover_Type column, and keep only class_label.
    self._sample = input_df.drop(columns=['Cover_Type'])
    self._description = self._load_readme(_README_FILE)

  @property
  def sample(self) -> pd.DataFrame:
    return self._sample

  @property
  def name(self) -> str:
    return _DATA_NAME

  @property
  def description(self) -> str:
    return self._description
