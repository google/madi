#     Copyright 2023 Google LLC
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
from madi.detectors import neg_sample_neural_net_detector as nsnn
from madi.utils import evaluation_utils
from madi.utils import sample_utils
import tensorflow as tf


class TestNegSampleNeuralNetDetector:

  def test_negative_sampling_dataset(self):
    """Tests that NegativeSamplingDataset has the correct sample ratio."""
    sample_ratio = 3.0
    ds = gaussian_mixture_dataset.GaussianMixtureDataset(
        n_dim=4,
        n_modes=1,
        n_pts_pos=2400,
        sample_ratio=sample_ratio,
        upper_bound=3,
        lower_bound=-3,
    ).sample
    normalization_info = sample_utils.get_normalization_info(ds)
    column_order = sample_utils.get_column_order(normalization_info)
    ns_ds = nsnn.NegativeSamplingDataset(
        sample_ratio=sample_ratio,
        batch_size=32,
        sample_delta=0.05,
        x_train=sample_utils.normalize(ds, normalization_info),
        column_order=column_order,
    )

    dataset = tf.data.Dataset.from_generator(
        ns_ds,
        output_signature=(
            tf.TensorSpec(shape=(None, len(column_order)), dtype=tf.float32),
            tf.TensorSpec(shape=(None), dtype=tf.float32),
        ),
    )

    def count_labels(counts, batch):
      labels = batch[1]
      for i in range(2):
        cc = tf.cast(labels == i, tf.int32)
        counts[i] += tf.reduce_sum(cc)
      return counts

    counts = dataset.reduce(
        initial_state={0: 0, 1: 0}, reduce_func=count_labels
    )
    assert counts[0] / counts[1] == sample_ratio

  def test_gaussian_mixture(self, tmpdir):
    """Tests NS-NN on single-mode Gaussian."""

    sample_ratio = 0.05
    ds = gaussian_mixture_dataset.GaussianMixtureDataset(
        n_dim=4,
        n_modes=1,
        n_pts_pos=2400,
        sample_ratio=sample_ratio,
        upper_bound=3,
        lower_bound=-3,
    )

    log_dir = tmpdir
    split_ix = int(len(ds.sample) * 0.8)
    training_sample = ds.sample.iloc[:split_ix]
    test_sample = ds.sample.iloc[split_ix:]

    ad = nsnn.NegativeSamplingNeuralNetworkAD(
        sample_ratio=3.0,
        sample_delta=0.05,
        batch_size=32,
        steps_per_epoch=16,
        epochs=20,
        dropout=0.5,
        learning_rate=0.001,
        layer_width=64,
        n_hidden_layers=2,
        patience=5,
        log_dir=log_dir)

    ad.train_model(x_train=training_sample.drop(columns=['class_label']))

    y_actual = test_sample['class_label']
    xy_predicted = ad.predict(test_sample.drop(columns=['class_label']))

    auc = evaluation_utils.compute_auc(
        y_actual=y_actual, y_predicted=xy_predicted['class_prob']
    )

    assert auc > 0.5

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
        lower_bound=-3,
    )

    log_dir = os.path.join(tmpdir, 'logs')
    split_ix = int(len(ds.sample) * 0.8)
    training_sample = ds.sample.iloc[:split_ix]
    test_sample = ds.sample.iloc[split_ix:]

    # Train a new model.
    ad_in = nsnn.NegativeSamplingNeuralNetworkAD(
        sample_ratio=3.0,
        sample_delta=0.05,
        batch_size=32,
        steps_per_epoch=16,
        epochs=20,
        dropout=0.5,
        learning_rate=0.001,
        layer_width=64,
        n_hidden_layers=2,
        patience=5,
        log_dir=log_dir)

    ad_in.train_model(x_train=training_sample.drop(columns=['class_label']))

    # Save the model to model_dir.
    ad_in.save_model(model_dir)

    # Create a new model with the same parameters.
    ad_out = nsnn.NegativeSamplingNeuralNetworkAD(
        sample_ratio=3.0,
        sample_delta=0.05,
        batch_size=32,
        steps_per_epoch=16,
        epochs=20,
        dropout=0.5,
        learning_rate=0.001,
        layer_width=64,
        n_hidden_layers=2,
        patience=5,
        log_dir=log_dir)
    # Load the previous trained and saved model.
    ad_out.load_model(model_dir)

    # Get some predictions and ensure the loaded model predicts accurately.
    y_actual = test_sample['class_label']
    xy_predicted = ad_out.predict(test_sample.drop(columns=['class_label']))

    auc = evaluation_utils.compute_auc(
        y_actual=y_actual, y_predicted=xy_predicted['class_prob'])

    assert auc > 0.5

  def test_early_stopping(self, tmpdir):
    """Tests that training stops early based on the validation accuracy."""

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
    num_epochs = 100

    ad = nsnn.NegativeSamplingNeuralNetworkAD(
        sample_ratio=3.0,
        sample_delta=0.05,
        batch_size=32,
        steps_per_epoch=16,
        epochs=num_epochs,
        dropout=0.5,
        learning_rate=0.001,
        layer_width=64,
        n_hidden_layers=2,
        patience=10,
        log_dir=log_dir)

    ad.train_model(x_train=training_sample.drop(columns=['class_label']))

    y_actual = test_sample['class_label']
    xy_predicted = ad.predict(test_sample.drop(columns=['class_label']))

    auc = evaluation_utils.compute_auc(
        y_actual=y_actual, y_predicted=xy_predicted['class_prob'])

    assert auc > 0.5
    assert len(ad.get_history().history['val_binary_accuracy']) < num_epochs
