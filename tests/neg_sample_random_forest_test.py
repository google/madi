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
"""Tests for madi.detectors.neg_sample_random_forest."""

from madi.datasets import gaussian_mixture_dataset
from madi.detectors.neg_sample_random_forest import NegativeSamplingRandomForestAd
import madi.utils.evaluation_utils as evaluation_utils


class TestNegSampleRandomForest:

  def test_gaussian_mixture(self):
    """Tests NS-NN on single-mode Gaussian."""

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

    num_estimators = 150
    criterion = 'gini'
    max_depth = 50
    min_samples_split = 12
    min_samples_leaf = 5
    min_weight_fraction_leaf = 0.06
    max_features = 0.3
    sample_ratio = 2.0
    sample_delta = 0.05

    ad = NegativeSamplingRandomForestAd(
        n_estimators=num_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=1,
        warm_start=False,
        class_weight=None,
        sample_delta=sample_delta,
        sample_ratio=sample_ratio)

    ad.train_model(x_train=training_sample.drop(columns=['class_label']))

    y_actual = test_sample['class_label']
    xy_predicted = ad.predict(test_sample.drop(columns=['class_label']))

    auc = evaluation_utils.compute_auc(
        y_actual=y_actual, y_predicted=xy_predicted['class_prob'])

    assert auc > 0.98
