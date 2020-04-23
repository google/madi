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
# Lint as: python3
"""Tests for google3.third_party.py.madi.utils.sample_utils."""

from absl.testing import absltest
import madi.utils.sample_utils as sample_utils
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal


class SampleUtilsTest(absltest.TestCase):

  def test_get_pos_sample_synthetic(self):

    n_points = 10000
    df = sample_utils.get_pos_sample_synthetic(
        mean=[0, 1, 2], cov=np.eye(3), n_points=n_points)
    self.assertLen(df, n_points)
    self.assertCountEqual(df.columns, ['class_label', 'x001', 'x002', 'x003'])
    self.assertAlmostEqual(np.mean(df['x001']), 0.0, 1)
    self.assertAlmostEqual(np.mean(df['x002']), 1.0, 1)
    self.assertAlmostEqual(np.mean(df['x003']), 2.0, 1)
    self.assertAlmostEqual(np.std(df['x001']), 1.0, 1)
    self.assertAlmostEqual(np.std(df['x002']), 1.0, 1)
    self.assertAlmostEqual(np.std(df['x003']), 1.0, 1)
    self.assertEqual(df['class_label'].all(), 1)

  def test_get_column_order(self):
    n_points = 10000
    df_in = sample_utils.get_pos_sample_synthetic(
        mean=[0, 1, 2], cov=np.eye(3), n_points=n_points)
    df_in = df_in.drop(columns=['class_label'])
    # Get the normalization info from the data frame.
    normalization_info = sample_utils.get_normalization_info(df_in)
    column_order = sample_utils.get_column_order(normalization_info)
    self.assertEqual(['x001', 'x002', 'x003'], column_order)

  def test_normalization_denormalization(self):
    n_points = 10000
    df_in = sample_utils.get_pos_sample_synthetic(
        mean=[0, 1, 2], cov=np.eye(3), n_points=n_points)
    df_in = df_in.drop(columns=['class_label'])
    # Get the normalization info from the data frame.
    normalization_info = sample_utils.get_normalization_info(df_in)
    # Apply the normalization info on the data frame.
    df_normalized = sample_utils.normalize(df_in, normalization_info)
    # Now, use the nor malization info to recover the original data.
    df_out = sample_utils.denormalize(df_normalized, normalization_info)
    # Finally, make sure that the data is perfectly recovered.
    assert_frame_equal(df_in, df_out)

  def test_get_neg_sample(self):
    n_points = 10000
    df_pos = pd.DataFrame({
        'x001': [1, 1, 1, -1, -1, -1],
        'x002': [0.5, 0.5, 0.5, -1.5, -1.5, -1.5],
        'x003': [110, 110, 110, 90, 90, 90],
        'class_label': 1
    })
    df_neg = sample_utils.get_neg_sample(
        df_pos, n_points, do_permute=False, delta=0.05)
    self.assertLen(df_neg, n_points)
    # dif = 20, 5% = 1: min should be one less, and max should be one more
    self.assertAlmostEqual(min(df_neg['x003']), 89, 1)
    self.assertAlmostEqual(max(df_neg['x003']), 111, 1)

  def test_get_positive_sample(self):
    df_raw = pd.DataFrame({
        'x001': [1, 1, 1, -1, -1, -1],
        'x002': [0.5, 0.5, 0.5, -1.5, -1.5, -1.5],
        'x003': [110, 110, 110, 90, 90, 90]
    })
    df_pos = sample_utils.get_pos_sample(df_raw, 3)
    self.assertLen(df_pos, 3)
    self.assertEqual(df_pos['class_label'].all(), 1)


if __name__ == '__main__':
  absltest.main()
