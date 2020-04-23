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
"""Tests for google3.third_party.py.madi.datasets.gaussian_dataset."""

from absl.testing import absltest
from madi.datasets import gaussian_mixture_dataset
import numpy as np


class GaussianMixtureDatasetTest(absltest.TestCase):

  def test_gaussian_dataset(self):
    n_dim = 4
    n_modes = 3
    n_pts_pos = 2400
    sample_ratio = 1.0
    n_pts_neg = int(sample_ratio * n_pts_pos)
    ds = gaussian_mixture_dataset.GaussianMixtureDataset(
        n_dim=n_dim,
        n_modes=n_modes,
        n_pts_pos=n_pts_pos,
        sample_ratio=sample_ratio,
        upper_bound=3,
        lower_bound=-3)

    self.assertLen(ds.sample, n_pts_neg + n_pts_pos)

    pos_sample = ds.sample[ds.sample['class_label'] == 1]
    for col in ['x001', 'x002', 'x003', 'x004']:
      # Verify that the three modes are present
      hist, _ = np.histogram(
          pos_sample[col], density=True, bins=12, range=(-4, 4))
      # Before mode 1:
      self.assertLessEqual(hist[0], 0.08)
      # First mode
      self.assertGreaterEqual(hist[2], 0.19)
      # Trough between modes 1 and 2
      self.assertLess(hist[4], 0.1)
      # Second mode
      self.assertGreaterEqual(min(hist[[5, 6]]), 0.16)
      # Trough between modes 2 and 3.
      self.assertLess(hist[7], 0.1)
      # Third mode
      self.assertGreaterEqual(hist[9], 0.19)
      # After mode 3
      self.assertLess(hist[11], 0.08)

    neg_sample = ds.sample[ds.sample['class_label'] == 0]
    for col in ['x001', 'x002', 'x003', 'x004']:
      # Verify that the distribution  is flat.
      hist, _ = np.histogram(
          neg_sample[col], density=True, bins=10, range=(-4, 4))

      self.assertLess(max(hist), 0.21)
      self.assertGreater(min(hist), 0.1)


if __name__ == '__main__':
  absltest.main()
