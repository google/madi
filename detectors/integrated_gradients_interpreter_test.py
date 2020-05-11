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
"""Tests for google3.third_party.py.madi.detectors.integrated_gradients_interpreter."""

import os

from absl.testing import absltest
import madi.detectors.integrated_gradients_interpreter
import madi.utils.sample_utils as sample_utils
import numpy as np
import numpy.testing
import pandas as pd
from pandas.util.testing import assert_frame_equal
import tensorflow as tf

_TEST_DATA = 'test_data'
_POSITIVE_SAMPLE_FILE = 'positive_sample.csv'
_BASELINE_FILE = 'baseline.csv'
_MODEL_FILENAME = 'model-multivariate-ad'

_TEST_ARRAY_1 = [
    .054484184936692, 1.4254372432567062, 2.4085036629196743,
    2.2868985633734993, 3.3217656144447694, 2.954329343296009,
    3.191659439569017, 3.09364593649725, 2.7713644613008954, 2.478310749742904,
    1.9968150428383753, 2.5516113664203663, 1.573991322768812,
    4.075890884631011, 2.5124624356597884, 3.115937849419807
]
_TEST_ARRAY_2 = [
    2.145085710132246, 2.3550537587508025, 2.602302820474312, 3.373158657628665,
    2.146149082864791, 2.186291117997812, 2.7779216471937223, 3.151762543822965,
    2.613204769667594, 2.1907670462347286, 2.6951671012676894,
    2.659408680645134, 2.7541670055236436, 3.1341381704681592,
    2.2923726556931228, 2.1051360078389982
]
_TEST_ARRAY_3 = [
    -1.7903824141571496, -1.1724005821041632, -1.4380521172756886,
    -1.681070953482498, -3.7799053794305513, -2.7943141775676157,
    -3.756951290155982, -2.247871312185007, -2.41788003208577,
    -2.2310721722925106, -3.1216753900574843, -1.6227496439852094,
    -2.8761174805422303, -2.722903450599994, -3.307142749666501,
    -2.0490122996462734
]
_TEST_ARRAY_4 = [
    3.4259646507765895, 1.7632137342252325, 3.0148266176722833,
    3.8551571412576884, 2.5526709080075505, 2.278832586149648,
    2.7602588190386967, 1.5980275416409906, 2.6439530522026136,
    2.456570536914754, 1.5430726289941117, 2.770272002968782,
    2.3422248954962197, 1.851925582594929, 3.463576278871484, 3.500698835736524
]

_FIELDS = [
    'x001', 'x002', 'x003', 'x004', 'x005', 'x006', 'x007', 'x008', 'x009',
    'x010', 'x011', 'x012', 'x013', 'x014', 'x015', 'x016'
]


class IntegratedGradientsInterpreterTest(absltest.TestCase):

  def test_blame(self):
    interpreter = self._get_test_interpreter(0.95, 250)
    anomalous_data = np.array(_TEST_ARRAY_4) + np.array(
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0])
    test_anomaly = pd.Series(data=anomalous_data, index=_FIELDS)
    actual_attribution_dict, actual_reference_point_dict, _ = interpreter.blame(
        test_anomaly)
    expected_attribution_dict = {
        'x001': 0.0,
        'x002': 0.0,
        'x003': 0.348,
        'x004': 0.0,
        'x005': 0.0,
        'x006': 0.0,
        'x007': 0.0,
        'x008': 0.0,
        'x009': 0.0,
        'x010': 0.0,
        'x011': 0.0,
        'x012': 0.0,
        'x013': 0.0,
        'x014': 0.0,
        'x015': 0.652,
        'x016': 0.0
    }
    expected_reference_point_dict = {
        'x001': 8.759257846003909,
        'x002': 4.52323561780342,
        'x003': 7.699828364950935,
        'x004': 9.779232288525447,
        'x005': 6.499157309002835,
        'x006': 5.824615852080057,
        'x007': 7.01870963226138,
        'x008': 4.1416220431627195,
        'x009': 6.729712284718885,
        'x010': 6.2612027775591965,
        'x011': 3.9207019605450446,
        'x012': 7.050307484508083,
        'x013': 6.017051998893703,
        'x014': 4.721141234703653,
        'x015': 8.854144729240176,
        'x016': 8.940268672582361
    }

    for col_name in expected_attribution_dict:
      self.assertAlmostEqual(actual_attribution_dict[col_name],
                             expected_attribution_dict[col_name], 3)
      self.assertAlmostEqual(actual_reference_point_dict[col_name],
                             expected_reference_point_dict[col_name], 3)

  def test_explain_1d(self):

    interpreter = self._get_test_interpreter(0.95, 250)
    reference = np.array(_TEST_ARRAY_1)
    sample = reference.copy()
    sample[4] = 10.0
    attribution, _ = interpreter.explain(
        sample=sample, reference=reference, num_steps=1000)

    numpy.testing.assert_array_almost_equal(
        np.abs(attribution), [
            0,
            0,
            0,
            0,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ],
        decimal=5)

  def test_explain_2d(self):

    interpreter = self._get_test_interpreter(0.95, 250)
    reference = np.array(_TEST_ARRAY_2)
    sample = reference.copy()
    sample[8] = 0
    sample[1] = 0
    attribution, _ = interpreter.explain(
        sample=sample, reference=reference, num_steps=1000)
    numpy.testing.assert_array_almost_equal(
        np.abs(attribution), [
            0.,
            0.72343098,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.27656902,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
            0.,
        ],
        decimal=5)

    reference = np.array(_TEST_ARRAY_3)
    sample = reference.copy()
    sample[0] = 4
    sample[7] = 6
    sample[14] = 7
    attribution, _ = interpreter.explain(
        sample=sample, reference=reference, num_steps=1000)
    numpy.testing.assert_array_almost_equal(
        np.abs(attribution), [
            0.23501, 0., 0., 0., 0., 0., 0., 0.44919, 0., 0., 0., 0., 0., 0.,
            0.3158, 0.
        ],
        decimal=5)

  def test_explain_3d(self):

    interpreter = self._get_test_interpreter(0.95, 250)
    reference = np.array(_TEST_ARRAY_3)
    sample = reference.copy()
    sample[0] = 4.
    sample[7] = 6.
    sample[14] = 7.
    attribution, _ = interpreter.explain(
        sample=sample, reference=reference, num_steps=1000)
    numpy.testing.assert_array_almost_equal(
        np.abs(attribution), [
            0.23501, 0., 0., 0., 0., 0., 0., 0.44919, 0., 0., 0., 0., 0., 0.,
            0.3158, 0.
        ],
        decimal=5)

  def test_find_nearest_euclidean(self):
    df_baseline = pd.read_csv(
        os.path.join(os.path.dirname(__file__), _TEST_DATA, _BASELINE_FILE),
        index_col=0)
    nearest_ix, nearest_dist = madi.detectors.integrated_gradients_interpreter.find_nearest_euclidean(
        df_baseline, np.array(_TEST_ARRAY_4))
    self.assertAlmostEqual(nearest_dist, 0.0, 5)
    self.assertEqual(nearest_ix, 4477)

    # Test with a small perturbation:
    nearest_ix, nearest_dist = madi.detectors.integrated_gradients_interpreter.find_nearest_euclidean(
        df_baseline,
        np.array(_TEST_ARRAY_4) + (0.01 * np.ones(16)))
    self.assertAlmostEqual(nearest_dist, 0.04, 5)
    self.assertEqual(nearest_ix, 4477)

  def test_select_invalid_baseline(self):

    with self.assertRaises(madi.detectors.integrated_gradients_interpreter
                           .NoQualifyingBaselineError) as cm:
      _ = self._get_test_interpreter(0.99, 1000)
    ex = cm.exception
    self.assertEqual(ex.min_class_confidence, 0.99)
    self.assertAlmostEqual(ex.highest_class_confidence, 0.97, places=2)

  def test_select_baseline(self):
    min_p = 0.95
    max_count = 100
    model_file_path = os.path.join(
        os.path.dirname(__file__), _TEST_DATA, _MODEL_FILENAME)
    model = tf.keras.models.load_model(model_file_path)
    df_pos = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), _TEST_DATA, _POSITIVE_SAMPLE_FILE),
        index_col=0)

    df_baseline_expected = pd.read_csv(
        os.path.join(os.path.dirname(__file__), _TEST_DATA, _BASELINE_FILE),
        index_col=0)

    df_baseline_actual, _ = madi.detectors.integrated_gradients_interpreter.select_baseline(
        df_pos=df_pos,
        model=model,
        normalization_info=sample_utils.get_normalization_info(df_pos),
        min_p=min_p,
        max_count=max_count)
    self.assertLen(df_baseline_actual, max_count)
    # Drop the probability column, loosen precison a little, and ignore actual
    # indicies.
    assert_frame_equal(
        df_baseline_actual.drop(columns=['class_prob']).reset_index(drop=True),
        df_baseline_expected.iloc[:max_count].reset_index(drop=True),
        check_like=True,
        check_exact=False,
        check_less_precise=3)
    self.assertGreaterEqual(min(df_baseline_actual['class_prob']), min_p)

  def _get_test_interpreter(self, min_class_confidence, max_baseline_size):
    df_pos = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), _TEST_DATA, _POSITIVE_SAMPLE_FILE),
        index_col=0)
    model_file_path = os.path.join(
        os.path.dirname(__file__), _TEST_DATA, _MODEL_FILENAME)
    model = tf.keras.models.load_model(model_file_path)

    # Load the previous trained and saved model.
    interpreter = madi.detectors.integrated_gradients_interpreter.IntegratedGradientsInterpreter(
        model, df_pos, sample_utils.get_normalization_info(df_pos),
        min_class_confidence, max_baseline_size)
    return interpreter


if __name__ == '__main__':
  absltest.main()
