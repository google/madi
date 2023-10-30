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
"""Tests for madi.detectors.one_class_svm."""


from madi.datasets import gaussian_mixture_dataset
from madi.detectors.one_class_svm import OneClassSVMAd
import madi.utils.evaluation_utils as evaluation_utils


class TestOneClassSvm:

  def test_gaussian_mixture(self):
    """Tests OC-SVM on single-mode Gaussian."""

    sample_ratio = 0.05
    ds = gaussian_mixture_dataset.GaussianMixtureDataset(
        n_dim=4,
        n_modes=1,
        n_pts_pos=2400,
        sample_ratio=sample_ratio,
        upper_bound=3,
        lower_bound=-3)

    split_ix = int(len(ds.sample) * 0.8)
    training_sample = ds.sample.iloc[:split_ix]
    test_sample = ds.sample.iloc[split_ix:]

    kernel = 'rbf'  # ["linear", "poly", "rbf", "sigmoid"]
    shrinking = True
    nu = 0.04  # Contamination
    gamma = 'scale'
    coef0 = 0.0

    ad = OneClassSVMAd(
        kernel=kernel,
        degree=3,
        gamma=gamma,
        coef0=coef0,
        tol=0.001,
        nu=nu,
        shrinking=shrinking,
        cache_size=200,
        verbose=False,
        max_iter=-1,
    )

    ad.train_model(x_train=training_sample.drop(columns=['class_label']))

    y_actual = test_sample['class_label']
    xy_predicted = ad.predict(test_sample.drop(columns=['class_label']))

    auc = evaluation_utils.compute_auc(
        y_actual=y_actual, y_predicted=xy_predicted['class_prob'])

    assert auc > 0.5
