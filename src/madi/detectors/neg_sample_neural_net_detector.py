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
"""Anomaly Detector based on Negative Sampling Neural Network (NS-NN)."""

from collections.abc import Sequence
import os
from typing import Optional
from absl import logging
from madi.detectors import base_detector
from madi.utils import sample_utils
import numpy as np
import pandas as pd
import tensorflow as tf

_TRAIN_VALIDATION_SPLIT = 0.8
_MODEL_FILENAME = 'model-multivariate-ad'
_NORMALIZATION_FILENAME = 'normalization_info'


class NegativeSamplingDataset:
  """A dataset with positive and negative samples in the given sample ratio."""

  def __init__(
      self,
      x_train: pd.DataFrame,
      sample_ratio: float,
      sample_delta: float,
      batch_size: int,
      column_order: Sequence[str],
  ):
    self._x_train = x_train.sample(frac=1)  # shuffles the data
    self._sample_ratio = sample_ratio
    self._sample_delta = sample_delta
    self._batch_size = batch_size
    self._column_order = column_order

  def __len__(self):
    return self._batch_size * (1 + self._sample_ratio)

  def __call__(self):
    mbatch = self._x_train.sample(n=self._batch_size, replace=True)
    training_sample = sample_utils.apply_negative_sample(
        positive_sample=mbatch,
        sample_ratio=self._sample_ratio,
        sample_delta=self._sample_delta,
    )
    x = np.float32(np.matrix(training_sample[self._column_order]))
    y = np.float32(np.array(training_sample['class_label']))
    yield x, y


class NegativeSamplingNeuralNetworkAD(
    base_detector.BaseAnomalyDetectionAlgorithm):
  """Anomaly detection using negative sampling and a neural net classifier."""

  def __init__(
      self,
      sample_ratio: float,
      sample_delta: float,
      batch_size: int,
      steps_per_epoch: int,
      epochs: int,
      dropout: float,
      learning_rate: float,
      layer_width: int,
      n_hidden_layers: int,
      patience: int,
      log_dir: str,
      tpu_worker: Optional[str] = None,
  ):
    self._sample_ratio = sample_ratio
    self._sample_delta = sample_delta
    self._model = None
    self._history = None
    self._steps_per_epoch = steps_per_epoch
    self._epochs = epochs
    self._dropout = dropout
    self._learning_rate = learning_rate
    self._layer_width = layer_width
    self._n_hidden_layers = n_hidden_layers
    self._log_dir = log_dir
    self._tpu_worker = tpu_worker
    self._batch_size = batch_size
    self._normalization_info = None
    self._patience = patience
    logging.info('TensorFlow version %s', tf.version.VERSION)

    # Especially with TPUs, it's useful to destroy the current TF graph and
    # creates a new one before getting started.
    tf.keras.backend.clear_session()

  def train_model(self, x_train: pd.DataFrame) -> None:
    """Train a new model and report the loss and accuracy.

    Args:
      x_train: dataframe with dimensions as columns.
    """
    self._normalization_info = sample_utils.get_normalization_info(x_train)
    column_order = sample_utils.get_column_order(self._normalization_info)
    normalized_x_train = sample_utils.normalize(x_train,
                                                self._normalization_info)
    train_points = int(_TRAIN_VALIDATION_SPLIT * len(normalized_x_train))

    train_ngen = NegativeSamplingDataset(
        x_train=normalized_x_train.iloc[:train_points],
        sample_ratio=self._sample_ratio,
        sample_delta=self._sample_delta,
        batch_size=self._batch_size,
        column_order=column_order,
    )
    train_dataset = tf.data.Dataset.from_generator(
        train_ngen,
        output_signature=(
            tf.TensorSpec(shape=(None, len(column_order)), dtype=tf.float32),
            tf.TensorSpec(shape=(None), dtype=tf.float32),
        ),
    ).repeat(self._batch_size * self._epochs)

    val_ngen = NegativeSamplingDataset(
        x_train=normalized_x_train.iloc[train_points:],
        sample_ratio=self._sample_ratio,
        sample_delta=self._sample_delta,
        batch_size=self._batch_size,
        column_order=column_order,
    )
    val_dataset = tf.data.Dataset.from_generator(
        val_ngen,
        output_signature=(
            tf.TensorSpec(shape=(None, len(column_order)), dtype=tf.float32),
            tf.TensorSpec(shape=(None), dtype=tf.float32)))

    if self._tpu_worker:
      resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
          self._tpu_worker
      )
      tf.contrib.distribute.initialize_tpu_system(resolver)
      strategy = tf.contrib.distribute.TPUStrategy(resolver)
      with strategy.scope():
        self._model = self._get_model(
            len(column_order),
            self._dropout,
            self._layer_width,
            self._n_hidden_layers,
        )
    else:
      self._model = self._get_model(
          len(column_order),
          self._dropout,
          self._layer_width,
          self._n_hidden_layers,
      )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_binary_accuracy', patience=self._patience)

    self._history = self._model.fit(
        x=train_dataset,
        steps_per_epoch=self._steps_per_epoch,
        verbose=1,
        epochs=self._epochs,
        validation_data=val_dataset,
        callbacks=[
            tf.keras.callbacks.TensorBoard(
                log_dir=self._log_dir,
                histogram_freq=1,
                write_graph=False,
                write_images=False,
            ),
            early_stopping,
        ],
    )

  def predict(self, sample_df: pd.DataFrame) -> pd.DataFrame:
    """Given new data, predict the probability of being positive class.

    Args:
      sample_df: DataFrame with features as columns, same as train().

    Returns:
      DataFrame with column 'class_prob' representing the probability of the
      Normal class.
    """

    sample_df_normalized = sample_utils.normalize(
        sample_df, self._normalization_info
    )
    column_order = sample_utils.get_column_order(self._normalization_info)
    x = np.float32(np.matrix(sample_df_normalized[column_order]))
    y_hat = self._model.predict(x, verbose=1, steps=1)
    sample_df['class_prob'] = y_hat
    return sample_df

  def get_history(self) -> tf.keras.callbacks.History:
    return self._history

  def _get_model(
      self,
      input_dim: int,
      dropout: float,
      layer_width: int,
      n_hidden_layers: int,
  ) -> tf.keras.Sequential:
    """Creates a Neural Network model for Anomaly Detection.

    Creates a simple stack of dense/dropout layers with equal width.

    Args:
      input_dim: width of the input layer
      dropout: dropout probability for each hidden layer
      layer_width: hidden layer width
      n_hidden_layers: number of hidden layers

    Returns:
      tf.keras.Sequential model.
    """
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Dense(
            layer_width, input_dim=input_dim, activation='relu'
        )
    )
    model.add(tf.keras.layers.Dropout(dropout))

    for _ in range(n_hidden_layers):
      model.add(tf.keras.layers.Dense(layer_width, activation='relu'))
      model.add(tf.keras.layers.Dropout(dropout))

    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=self._learning_rate,
        rho=0.9,
        momentum=0.0,
        epsilon=1e-07,
        centered=False,
        weight_decay=None,
        clipnorm=None,
        clipvalue=None,
        global_clipnorm=None,
        use_ema=False,
        ema_momentum=0.99,
        ema_overwrite_frequency=100,
        jit_compile=True,
        name='RMSprop',
    )
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizer,
        metrics=[tf.keras.metrics.binary_accuracy],
    )
    return model

  def save_model(self, model_dir: str) -> None:
    """Saves the trained AD model to the model directory model_dir."""
    model_file_path = os.path.join(model_dir, _MODEL_FILENAME)
    tf.keras.models.save_model(self._model, model_file_path, overwrite=True)
    logging.info('Sucessfully wrote the model to %s', model_file_path)
    normalization_file_path = os.path.join(model_dir, _NORMALIZATION_FILENAME)
    sample_utils.write_normalization_info(self._normalization_info,
                                          normalization_file_path)
    logging.info('Successfully saved normalization info to %s',
                 normalization_file_path)

  def load_model(self, model_dir: str) -> None:
    """Loads the trained AD model from the model directory model_dir."""
    model_file_path = os.path.join(model_dir, _MODEL_FILENAME)
    self._model = tf.keras.models.load_model(model_file_path)
    logging.info('Successfully loaded model from %s', model_file_path)
    normalization_file_path = os.path.join(model_dir, _NORMALIZATION_FILENAME)
    self._normalization_info = sample_utils.read_normalization_info(
        normalization_file_path)
    logging.info('Successfully read normalization info from %s',
                 normalization_file_path)
