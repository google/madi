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
"""Dataset generator for multimodal, multidimensional Gaussian data."""

from typing import Optional, Iterable

from madi.datasets.base_dataset import BaseDataset
import numpy as np
import pandas as pd


class GaussianMixtureDataset(BaseDataset):
  """Generates a multimodal, multidimensional dataset for Anomaly Detection."""

  def __init__(self, n_dim: int, n_modes: int, n_pts_pos: int,
               sample_ratio: float, upper_bound: float, lower_bound: float):
    self._n_dim = n_dim
    self._n_modes = n_modes
    self._n_pts_pos = n_pts_pos
    self._sample_ratio = sample_ratio
    self._upper_bound = upper_bound
    self._lower_bound = lower_bound
    self._sample = self._get_mdim_gaussian_sample(
        n_pts_pos=self._n_pts_pos,
        n_dim=self._n_dim,
        sample_ratio=self._sample_ratio,
        noise_dim=None,
        n_modes=self._n_modes,
        neg_min=self._lower_bound,
        neg_max=self._upper_bound)

  @property
  def sample(self) -> pd.DataFrame:
    return self._sample

  @property
  def name(self) -> str:
    return "gaussian"

  @property
  def description(self) -> str:
    return ("{dim}-Dimensional, {modes}-Modal Gaussian distribution with sample"
            " ratio = {sample_ratio} and {n_points} sample points.").format(
                dim=self._n_dim,
                modes=self._modes,
                sample_ratio=self._sample_ratio,
                n_points=len(self._sample))

  def _get_mdim_gaussian_sample(self,
                                n_pts_pos: int,
                                n_dim: int,
                                sample_ratio: float,
                                noise_dim: Optional[Iterable[int]] = None,
                                n_modes: int = 1,
                                neg_min: float = -3,
                                neg_max: float = 3) -> pd.DataFrame:
    """Generates a multidimensional Gaussian synthetic test set.

    Args:
      n_pts_pos: number of positive sample points.
      n_dim: number of dimensions
      sample_ratio: proportion of negative sample size to positive
      noise_dim: array of dimensions to add uniform noise
      n_modes: number of modes distributed along x001 = x002 = ... = x[d]
      neg_min: minimum mode postion
      neg_max: maximum mode position

    Returns:
      a shuffled sample dataframe, of  d-dim, n-points and class label.
    """

    n_pts_neg = int(n_pts_pos * sample_ratio)

    def _get_pos_sample_synthetic(mean: float, cov: float,
                                  n_points: int) -> pd.DataFrame:
      """Generates a positive sample from a Gaussian distribution with n_points.

      Args:
        mean: d-dimensional vector of mean values.
        cov: dxd dimensional covariance matrix.
        n_points: Number of points to return.

      Returns:
        DataFrame with cols x001...x[d] and n_points rows drawn from Guassian
        with
        mean and cov.
      """

      pos_mat = np.random.multivariate_normal(mean, cov, n_points).T
      df_pos = pd.DataFrame({"class_label": [1 for _ in range(n_points)]})

      for i in range(pos_mat.shape[0]):
        df_pos["x%03d" % (i + 1)] = pos_mat[i]
      return df_pos

    def get_multidim_gaussian(n_points, n_dim, meanv=0, varv=1):
      cov = np.identity(n_dim) * varv
      mean_vec = np.ones(n_dim) * meanv

      return _get_pos_sample_synthetic(mean_vec, cov, n_points)

    def get_uniform_sample(n_points, n_dim, min_val, max_val):
      s = np.random.uniform(min_val, max_val, n_points * n_dim).reshape(
          (n_points, n_dim))
      cols = ["x%03d" % (1 + i) for i in range(n_dim)]
      neg = pd.DataFrame(s, columns=cols)
      neg["class_label"] = 0
      return neg

    varv = 1 / float(n_modes)
    # With one mode, just place the gaussian at the origin.
    if n_modes == 1:
      sample = get_multidim_gaussian(
          n_points=n_pts_pos, n_dim=n_dim, meanv=0, varv=varv)

    else:
      sample = pd.DataFrame()
      n_pts_msample = int(n_pts_pos / n_modes)
      # Want to resize the modes so that they (a) separate reasonably and (b)
      # yield sufficiently sparse regions between modes while still being
      # contained (mostly) within the neg_min and neg_max. The factor 0.8 wa
      # chosen because it enables 2 - 8 modes to meert those criteria.
      h = 0.8 * (neg_max - neg_min) / (n_modes - 1)
      for m in range(n_modes):
        meanv = neg_min * 0.8 + float(m) * h
        msample = get_multidim_gaussian(
            n_points=n_pts_msample, n_dim=n_dim, meanv=meanv, varv=varv)
        sample = pd.concat([sample, msample])

    if noise_dim:
      for index_id in noise_dim:
        sample[index_id] = np.random.uniform(neg_min, neg_max, len(sample))

    if n_pts_neg > 0:

      neg_sample = get_uniform_sample(
          n_pts_neg, n_dim=n_dim, min_val=neg_min * 2, max_val=neg_max * 2)

      sample = pd.concat([sample, neg_sample], ignore_index=True, sort=True)

    return sample.reindex(np.random.permutation(sample.index))


