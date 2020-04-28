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
"""Anomaly Detector based on Negative Sampling Neural Network (NS-NN)."""

from absl import logging
from madi.detectors.base_detector import BaseAnomalyDetectionAlgorithm
from madi.detectors.base_interpreter import BaseAnomalyInterpreter
import madi.utils.sample_utils as sample_utils
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

_SHUFFLE_BUFFERSIZE = 500


class NegativeSamplingNeuralNetworkAD(BaseAnomalyDetectionAlgorithm,
                                      BaseAnomalyInterpreter):
  """Anomaly detection using negative sampling and a neural net classifier."""

  def __init__(self,
               sample_ratio: float,
               sample_delta: float,
               batch_size: int,
               steps_per_epoch: int,
               epochs: int,
               dropout: float,
               layer_width: int,
               n_hidden_layers: int,
               log_dir: str,
               tpu_worker: str = None):
    self._sample_ratio = sample_ratio
    self._sample_delta = sample_delta
    self._model = None
    self._history = None
    self._steps_per_epoch = steps_per_epoch
    self._epochs = epochs
    self._dropout = dropout
    self._layer_width = layer_width
    self._n_hidden_layers = n_hidden_layers
    self._log_dir = log_dir
    self._tpu_worker = tpu_worker
    self._batch_size = batch_size
    logging.info('TensorFlow version %s', tf.version.VERSION)

    # Especially with TPUs, it's useful to destroy the current TF graph and
    # creates a new one before getting started.
    tf.keras.backend.clear_session()

  def train_model(self, x_train: pd.DataFrame) -> None:
    """Train a new model and report the loss and accuracy.

    Args:
      x_train: dataframe with dimensions as columns.
    """
    training_sample = sample_utils.apply_negative_sample(
        positive_sample=x_train,
        sample_ratio=self._sample_ratio,
        sample_delta=self._sample_delta)

    x = np.float32(np.matrix(training_sample.drop(columns=['class_label'])))
    y = np.float32(np.array(training_sample['class_label']))
    # create dataset objects from the arrays
    dx = tf.data.Dataset.from_tensor_slices(x)
    dy = tf.data.Dataset.from_tensor_slices(y)

    logging.info('Training ns-nn with:')
    logging.info(training_sample['class_label'].value_counts())

    # zip the two datasets together
    train_dataset = tf.data.Dataset.zip(
        (dx, dy)).shuffle(_SHUFFLE_BUFFERSIZE).repeat().batch(self._batch_size)

    if self._tpu_worker:
      resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          self._tpu_worker)
      tf.contrib.distribute.initialize_tpu_system(resolver)
      strategy = tf.contrib.distribute.TPUStrategy(resolver)
      with strategy.scope():
        self._model = self._get_model(x.shape[1], self._dropout,
                                      self._layer_width, self._n_hidden_layers)
    else:
      self._model = self._get_model(x.shape[1], self._dropout,
                                    self._layer_width, self._n_hidden_layers)

    self._model.fit(
        x=train_dataset,
        steps_per_epoch=self._steps_per_epoch,
        verbose=0,
        epochs=self._epochs,
        callbacks=[
            keras.callbacks.TensorBoard(
                log_dir=self._log_dir,
                histogram_freq=1,
                write_graph=False,
                write_images=False)
        ])

  def predict(self, sample_df: pd.DataFrame) -> pd.DataFrame:
    """Given new data, predict the probability of being positive class.

    Args:
      sample_df: dataframe with features as columns, same as train().

    Returns:
      DataFrame as sample_df, with colum 'class_prob', prob of Normal class.
    """

    x = np.float32(np.matrix(sample_df))
    y_hat = self._model.predict(x, verbose=1, steps=1)
    sample_df['class_prob'] = y_hat
    return sample_df

  def _get_model(self, input_dim, dropout, layer_width, n_hidden_layers):
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(layer_width, input_dim=input_dim, activation='relu'))
    model.add(keras.layers.Dropout(dropout))

    for _ in range(n_hidden_layers):
      model.add(keras.layers.Dense(layer_width, activation='relu'))
      model.add(keras.layers.Dropout(dropout))

    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='rmsprop',
        metrics=[tf.keras.metrics.binary_accuracy])
    return model

  def blame(self, anomaly: np.array) -> np.array:
    """Provides interpreration using integrated gradients."""
    return None
