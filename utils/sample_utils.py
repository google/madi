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
"""Utilities to to generate or modify data samples."""

from typing import List, Dict

import numpy as np
import pandas as pd
import tensorflow as tf


class Variable(object):

  def __init__(self, index, name, mean, std):
    self.index = index
    self.mean = mean
    self.name = name
    self.std = std


def get_normalization_info(df: pd.DataFrame) -> Dict[str, Variable]:
  """Computes means, standard deviation to normalize a data frame.

  Any variable xxxx_validity is considered a boolean validity indicator
  for variable xxxx, and will not be normalized. A value of 1
  indicates the value xxxx is valid, and 0 indicates xxx is invalid.

  Args:
    df: Pandas dataframe with numeric feature data.

  Returns:
    A dict with Variable.name, Variable.
  """
  variables = {}
  for column in df:
    if not np.issubdtype(df[column].dtype, np.number):
      raise ValueError("The feature column %s is not numeric." % column)

    if column.endswith("_validity"):
      vmean = 0.0
      vstd = 1.0
    else:
      vmean = df[column].mean()
      vstd = df[column].std()

    variable = Variable(
        index=df.columns.get_loc(column),
        name=column,
        mean=vmean,
        std=vstd)
    variables[column] = variable
  return variables


def get_column_order(normalization_info: Dict[str, Variable]) -> List[str]:
  """Returns a list of column names, as strings, in model order."""
  return [
      var.name
      for var in sorted(normalization_info.values(), key=lambda var: var.index)
  ]


def normalize(df: pd.DataFrame,
              normalization_info: Dict[str, Variable]) -> pd.DataFrame:
  """Normalizes an input Dataframe of features.

  Args:
    df: Pandas DataFrame of M rows with N real-valued features
    normalization_info: dict of name, variable types containing mean, and std.

  Returns:
    Pandas M x N DataFrame with normalized features.
  """
  df_norm = pd.DataFrame()
  for column in get_column_order(normalization_info):
    df_norm[column] = (df[column] - normalization_info[column].mean
                      ) / normalization_info[column].std
  return df_norm


def denormalize(df_norm: pd.DataFrame,
                normalization_info: Dict[str, Variable]) -> pd.DataFrame:
  """Reverts normalization an input Dataframe of features.

  Args:
    df_norm: Pandas DataFrame of M rows with N real-valued normalized features
    normalization_info: dict of name, variable types containing mean, and std.

  Returns:
    Pandas M x N DataFrame with denormalized features.
  """
  df = pd.DataFrame()
  for column in get_column_order(normalization_info):
    df[column] = df_norm[column] * normalization_info[
        column].std + normalization_info[column].mean
  return df


def write_normalization_info(normalization_info: Dict[str, Variable],
                             filename: str):
  """Writes variable normalization info to CSV."""

  def to_df(normalization_info):
    df = pd.DataFrame(columns=["index", "mean", "std"])
    for variable in normalization_info:
      df.loc[variable] = [
          normalization_info[variable].index, normalization_info[variable].mean,
          normalization_info[variable].std
      ]
    return df

  with tf.io.gfile.GFile(filename, "w") as csv_file:
    to_df(normalization_info).to_csv(csv_file, sep="\t")


def read_normalization_info(
    filename: str) -> Dict[str, Variable]:
  """Reads variable normalization info from CSV."""

  def from_df(df):
    normalization_info = {}
    for name, row in df.iterrows():
      normalization_info[name] = Variable(
          row["index"], name, row["mean"], row["std"])
    return normalization_info

  normalization_info = {}
  if not tf.io.gfile.exists(filename):
    raise AssertionError("{} does not exist".format(filename))
  with tf.io.gfile.GFile(filename, "r") as csv_file:
    df = pd.read_csv(csv_file, header=0, index_col=0, sep="\t")
    normalization_info = from_df(df)
  return normalization_info


def get_neg_sample(pos_sample: pd.DataFrame,
                   n_points: int,
                   do_permute: bool = False,
                   delta: float = 0.0) -> pd.DataFrame:
  """Creates a negative sample from the cuboid bounded by +/- delta.

  Where, [min - delta, max + delta] for each of the dimensions.
  If do_permute, then rather than uniformly sampling, simply
  randomly permute each dimension independently.
  The positive sample, pos_sample is a pandas DF that has a column
  labeled 'class_label' where 1.0 indicates Normal, and
  0.0 indicates anomalous.

  Args:
    pos_sample: DF with numeric dimensions
    n_points: number points to be returned
    do_permute: permute or sample
    delta: fraction of [max - min] to extend the sampling.

  Returns:
    A dataframe  with the same number of columns, and a label column
    'class_label' where every point is 0.
  """
  df_neg = pd.DataFrame()

  pos_sample_n = pos_sample.sample(n=n_points, replace=True)

  for field_name in list(pos_sample):

    if field_name == "class_label":
      continue

    if do_permute:
      df_neg[field_name] = np.random.permutation(
          np.array(pos_sample_n[field_name]))

    else:
      low_val = min(pos_sample[field_name])
      high_val = max(pos_sample[field_name])
      delta_val = high_val - low_val
      df_neg[field_name] = np.random.uniform(
          low=low_val - delta * delta_val,
          high=high_val + delta * delta_val,
          size=n_points)

  df_neg["class_label"] = [0 for _ in range(n_points)]
  return df_neg


def apply_negative_sample(positive_sample: pd.DataFrame, sample_ratio: float,
                          sample_delta: float) -> pd.DataFrame:
  """Returns a dataset with negative and positive sample.

  Args:
    positive_sample: actual, observed sample where each col is a feature.
    sample_ratio: the desired ratio of negative to positive points
    sample_delta: the extension beyond observed limits to bound the neg sample

  Returns:
    DataFrame with features + class label, with 1 being observed and 0 negative.
  """

  positive_sample["class_label"] = 1
  n_neg_points = int(len(positive_sample) * sample_ratio)
  negative_sample = get_neg_sample(
      positive_sample, n_neg_points, do_permute=False, delta=sample_delta)
  training_sample = pd.concat([positive_sample, negative_sample],
                              ignore_index=True,
                              sort=True)
  return training_sample.reindex(np.random.permutation(training_sample.index))


def get_pos_sample(df_input: pd.DataFrame, n_points: int) -> pd.DataFrame:
  """Draws n_points from the data sample, and adds a class_label column."""
  df_pos = df_input.sample(n=n_points)
  df_pos["class_label"] = 1
  return df_pos


def get_train_data(input_df: pd.DataFrame,
                   n_points: int,
                   sample_ratio: float = 1.0,
                   do_permute: bool = True):
  """Generates a test and train data set for buidlings a test model.

  Args:
    input_df: dataframe containing observed, real-valued data, where each field
      is a dimension.
    n_points: total number points to be returned (positive and negative)
    sample_ratio: rtio, neg sample / pos sample sizes (e.g., 2 = means 2x neg
      points as pos)
    do_permute: False, uniformly sample; True. sample positive and permute cols

  Returns:
    x: Dataframe with d-Dim cols and, n_points rows
    y: class labels, with 1 = Normal/positive and 0 = Anomalous/negative class
  """

  # Create the positive class sample, with mean at the origin and a
  # rotated covariance matrix.
  n_pos = int(n_points / (sample_ratio + 1.0))
  n_neg = n_points - n_pos
  # Gather a random subsample of length n, as the positive set.
  df_pos = get_pos_sample(input_df, n_pos)

  if sample_ratio > 0.0:
    # Generate a random negative sample.
    df_neg = get_neg_sample(df_pos, n_neg, do_permute)

    # Combine both, randomize and split.
    df_combined = pd.concat([df_pos, df_neg], ignore_index=True)
  else:
    df_combined = df_pos.sample(n=n_points)
  df_combined = df_combined.iloc[np.random.permutation(len(df_combined))]

  # Separate the labels, and remove the column.
  y = df_combined["class_label"]
  x = df_combined.drop(columns=["class_label"])
  return x, y


def get_pos_sample_synthetic(mean: float, cov: float,
                             n_points: int) -> pd.DataFrame:
  """Generates a positive sample from a Gaussian distribution with n_points.

  Args:
    mean: d-dimensional vector of mean values.
    cov: dxd dimensional covariance matrix.
    n_points: Number of points to return.

  Returns:
    DataFrame with cols x001...x[d] and n_points rows drawn from Guassian with
    mean and cov.
  """

  pos_mat = np.random.multivariate_normal(mean, cov, n_points).T
  df_pos = pd.DataFrame({"class_label": [1 for _ in range(n_points)]})

  for i in range(pos_mat.shape[0]):
    df_pos["x%03d" % (i + 1)] = pos_mat[i]
  return df_pos
