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
"""Tests for madi.utils.sample_utils."""

import madi.utils.sample_utils as sample_utils
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
from pandas.util.testing import assert_series_equal
import pytest


class TestSampleUtilsTest:

  def test_get_pos_sample_synthetic(self):

    n_points = 10000
    df = sample_utils.get_pos_sample_synthetic(
        mean=[0, 1, 2], cov=np.eye(3), n_points=n_points)
    assert len(df) == n_points
    assert sorted(df.columns) == sorted(['class_label', 'x001', 'x002', 'x003'])

    assert np.mean(df['x001']) == pytest.approx(0.0, abs=1e-1)
    assert np.mean(df['x002']) == pytest.approx(1.0, abs=1e-1)
    assert np.mean(df['x003']) == pytest.approx(2.0, abs=1e-1)
    assert np.std(df['x001']) == pytest.approx(1.0, abs=1e-1)
    assert np.std(df['x002']) == pytest.approx(1.0, abs=1e-1)
    assert np.std(df['x003']) == pytest.approx(1.0, abs=1e-1)
    assert df['class_label'].all()

  def test_normalize_with_validity_indicator(self):

    n_points = 10000
    df_in = sample_utils.get_pos_sample_synthetic(
        mean=[0, 1, 2], cov=np.eye(3), n_points=n_points)
    df_in['x003_validity'] = np.concatenate(
        (np.ones(n_points - 100), np.zeros(100)), axis=0)
    normalization_info = sample_utils.get_normalization_info(
        df_in.drop(columns=['class_label']))

    df_normalized = sample_utils.normalize(df_in, normalization_info)
    assert sorted(normalization_info.keys()) == sorted(
        ['x001', 'x002', 'x003', 'x003_validity'])

    assert_series_equal(df_in['x003_validity'], df_normalized['x003_validity'])

  def test_get_column_order(self):
    n_points = 10000
    df_in = sample_utils.get_pos_sample_synthetic(
        mean=[0, 1, 2], cov=np.eye(3), n_points=n_points)
    df_in = df_in.drop(columns=['class_label'])
    # Get the normalization info from the data frame.
    normalization_info = sample_utils.get_normalization_info(df_in)
    column_order = sample_utils.get_column_order(normalization_info)
    assert column_order == ['x001', 'x002', 'x003']

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
    assert len(df_neg) == n_points
    # dif = 20, 5% = 1: min should be one less, and max should be one more
    assert min(df_neg['x003']) == pytest.approx(89, abs=1e-1)
    assert max(df_neg['x003']) == pytest.approx(111, abs=1e-1)

  def test_get_positive_sample(self):
    df_raw = pd.DataFrame({
        'x001': [1, 1, 1, -1, -1, -1],
        'x002': [0.5, 0.5, 0.5, -1.5, -1.5, -1.5],
        'x003': [110, 110, 110, 90, 90, 90]
    })
    df_pos = sample_utils.get_pos_sample(df_raw, 3)
    assert len(df_pos) == 3
    assert df_pos['class_label'].all()

  def test_apply_negative_sample(self):

    positive_sample = pd.DataFrame({
        'x001': [-1.0, 0.0, -0.4, -0.5, -1.0],
        'x002': [67.0, 50.0, 98.0, 100.0, 77.0],
        'x003': [0.0001, 0.00011, 0.0008, 0.0009, 0.0005]
    })
    sample = sample_utils.apply_negative_sample(positive_sample, 10, 0.05)
    assert_series_equal(sample['class_label'].value_counts(),
                        pd.Series([50, 5], name='class_label', index=[0, 1]))
    assert min(sample['x001']) >= -1.05
    assert max(sample['x001']) <= 0.05
    assert min(sample['x002']) >= 47.5
    assert max(sample['x002']) <= 102.5
    assert min(sample['x003']) >= 6e-05
    assert max(sample['x003']) <= 0.00094
