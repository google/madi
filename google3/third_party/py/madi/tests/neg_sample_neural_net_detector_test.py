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
"""Tests for madi.detectors.neg_sample_neural_net_detector."""
import os

from madi.datasets import gaussian_mixture_dataset
from madi.detectors.neg_sample_neural_net_detector import NegativeSamplingNeuralNetworkAD
import madi.utils.evaluation_utils as evaluation_utils


class TestNegSampleNeuralNetDetector:

  def test_gaussian_mixture(self, tmpdir):
    """Tests NS-NN on single-mode Gaussian."""

    sample_ratio = 0.05
    ds = gaussian_mixture_dataset.GaussianMixtureDataset(
        n_dim=4,
        n_modes=1,
        n_pts_pos=2400,
        sample_ratio=sample_ratio,
        upper_bound=3,
        lower_bound=-3)

    log_dir = tmpdir
    split_ix = int(len(ds.sample) * 0.8)
    training_sample = ds.sample.iloc[:split_ix]
    test_sample = ds.sample.iloc[split_ix:]

    ad = NegativeSamplingNeuralNetworkAD(
        sample_ratio=3.0,
        sample_delta=0.05,
        batch_size=32,
        steps_per_epoch=16,
        epochs=20,
        dropout=0.5,
        layer_width=64,
        n_hidden_layers=2,
        log_dir=log_dir)

    ad.train_model(x_train=training_sample.drop(columns=['class_label']))

    y_actual = test_sample['class_label']
    xy_predicted = ad.predict(test_sample.drop(columns=['class_label']))

    auc = evaluation_utils.compute_auc(
        y_actual=y_actual, y_predicted=xy_predicted['class_prob'])

    assert auc > 0.98

  def test_gaussian_mixture_io(self, tmpdir):
    """Tests NS-NN on single-mode Gaussian."""

    model_dir = os.path.join(tmpdir, 'models')
    sample_ratio = 0.05
    ds = gaussian_mixture_dataset.GaussianMixtureDataset(
        n_dim=4,
        n_modes=1,
        n_pts_pos=2400,
        sample_ratio=sample_ratio,
        upper_bound=3,
        lower_bound=-3)

    log_dir = os.path.join(tmpdir, 'logs')
    split_ix = int(len(ds.sample) * 0.8)
    training_sample = ds.sample.iloc[:split_ix]
    test_sample = ds.sample.iloc[split_ix:]

    # Train a new model.
    ad_in = NegativeSamplingNeuralNetworkAD(
        sample_ratio=3.0,
        sample_delta=0.05,
        batch_size=32,
        steps_per_epoch=16,
        epochs=20,
        dropout=0.5,
        layer_width=64,
        n_hidden_layers=2,
        log_dir=log_dir)

    ad_in.train_model(x_train=training_sample.drop(columns=['class_label']))

    # Save the model to model_dir.
    ad_in.save_model(model_dir)

    # Create a new model with the same parameters.
    ad_out = NegativeSamplingNeuralNetworkAD(
        sample_ratio=3.0,
        sample_delta=0.05,
        batch_size=32,
        steps_per_epoch=16,
        epochs=20,
        dropout=0.5,
        layer_width=64,
        n_hidden_layers=2,
        log_dir=log_dir)
    # Load the previous trained and saved model.
    ad_out.load_model(model_dir)

    # Get some predictions and ensure the loaded model predicts accurately.
    y_actual = test_sample['class_label']
    xy_predicted = ad_out.predict(test_sample.drop(columns=['class_label']))

    auc = evaluation_utils.compute_auc(
        y_actual=y_actual, y_predicted=xy_predicted['class_prob'])

    assert auc > 0.98
