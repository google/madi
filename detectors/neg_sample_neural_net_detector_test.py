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
"""Tests for google3.third_party.py.madi.detectors.neg_sample_neural_net_detector."""

from absl.testing import absltest
from madi.datasets import gaussian_mixture_dataset
from madi.detectors.neg_sample_neural_net_detector import NegativeSamplingNeuralNetworkAD
import madi.utils.evaluation_utils as evaluation_utils


class NegSampleNeuralNetDetectorTest(absltest.TestCase):

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

    log_dir = self.create_tempdir()
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

    self.assertGreater(auc, 0.99)


if __name__ == '__main__':
  absltest.main()
