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
"""Tests for madi.detectors.integrated_gradients_interpreter."""

import os

import madi.detectors.integrated_gradients_interpreter
import madi.utils.sample_utils as sample_utils
import numpy as np
import numpy.testing
import pandas as pd
import pytest
import tensorflow as tf

_TEST_DATA = os.path.join(os.path.dirname(__file__), 'test_data')
_POSITIVE_SAMPLE_FILE = 'positive_sample.csv'
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


class TestIntegratedGradientsInterpreter:

  def test_blame(self):
    interpreter = self._get_test_interpreter(0.95, 250)
    anomalous_data = np.array(_TEST_ARRAY_4) + np.array(
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0])
    test_anomaly = pd.Series(data=anomalous_data, index=_FIELDS)
    actual_attribution_dict, actual_reference_point_dict, _ = interpreter.blame(
        test_anomaly)

    expected_attribution_dict = {
        'x001': 0.002187757633945183,
        'x002': 0.002976363240582482,
        'x003': 0.2646628511083332,
        'x004': 0.03219711570075021,
        'x005': 0.013916306551774987,
        'x006': 0.02314098749698794,
        'x007': 0.010694018796641655,
        'x008': 0.021948330285941772,
        'x009': 0.011765977701929463,
        'x010': 0.013192013274398609,
        'x011': 0.023741249534000843,
        'x012': 0.05950213360248674,
        'x013': 0.026055538813884584,
        'x014': 0.016518005104400192,
        'x015': 0.46945848858672456,
        'x016': 0.008042862567217661
    }
    expected_reference_point_dict = {
        'x001': 1.0633042313064187,
        'x002': 1.051456808162845,
        'x003': 1.3846602854823593,
        'x004': 1.0978800623570324,
        'x005': 1.4325814172537334,
        'x006': 1.1697147545439555,
        'x007': 0.5788864804073347,
        'x008': 1.2042111655483745,
        'x009': 1.1065457242952912,
        'x010': 0.9953755474877946,
        'x011': 0.9089955288877173,
        'x012': 1.0400328217614323,
        'x013': 1.23407202478703,
        'x014': 1.2931078015872917,
        'x015': 1.5710934132687078,
        'x016': 1.0148944337105463
    }
    for col_name in expected_attribution_dict:
      assert actual_attribution_dict[col_name] == pytest.approx(
          expected_attribution_dict[col_name], abs=1e-5)
      assert actual_reference_point_dict[col_name] == pytest.approx(
          expected_reference_point_dict[col_name], abs=1e-5)

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
        os.path.join(
            os.path.dirname(__file__), _TEST_DATA, _POSITIVE_SAMPLE_FILE),
        index_col=0)
    nearest_ix, nearest_dist = madi.detectors.integrated_gradients_interpreter.find_nearest_euclidean(
        df_baseline, np.array(_TEST_ARRAY_4))
    assert nearest_dist == pytest.approx(0.0, abs=1e-5)
    assert nearest_ix == 4477

    # Test with a small perturbation:
    nearest_ix, nearest_dist = madi.detectors.integrated_gradients_interpreter.find_nearest_euclidean(
        df_baseline,
        np.array(_TEST_ARRAY_4) + (0.01 * np.ones(16)))
    assert nearest_dist == pytest.approx(0.04, abs=1e-5)
    assert nearest_ix == 4477

  def test_select_invalid_baseline(self):

    with pytest.raises(madi.detectors.integrated_gradients_interpreter
                       .NoQualifyingBaselineError) as cm:
      _ = self._get_test_interpreter(0.99, 1000)
    ex = cm.value
    assert ex.min_class_confidence == 0.99
    assert ex.highest_class_confidence == pytest.approx(0.97, abs=1e-2)

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

    df_baseline_actual, max_cnf = madi.detectors.integrated_gradients_interpreter.select_baseline(
        df_pos_normalized=df_pos, model=model, min_p=min_p, max_count=max_count)

    expected_indices = [
        290, 1041, 1482, 207, 571, 640, 3090, 830, 2088, 1176, 1722, 1551, 2023,
        1701, 2186, 83, 943, 3495, 656, 548, 2261, 1086, 1588, 1829, 3742, 1406,
        11, 2230, 536, 2162, 124, 2436, 864, 756, 1431, 33, 1862, 996, 2132,
        1646, 1944, 831, 1358, 2284, 2963, 414, 773, 1134, 1835, 1391, 386,
        2434, 673, 403, 512, 2847, 1186, 4099, 41, 694, 591, 1344, 412, 1772,
        1102, 1519, 2142, 165, 3268, 560, 1329, 4743, 4481, 2297, 610, 1528, 64,
        2305, 1197, 2371, 335, 1225, 1892, 2926, 1417, 411, 2383, 806, 1441,
        375, 152, 2183, 3136, 510, 1459, 3108, 1242, 1049, 1389, 342
    ]
    assert df_baseline_actual.index.tolist() == expected_indices
    assert max_cnf >= min_p

  def test_completeness_axiom_on_large_perturbation(self):
    """Tests the Completeness Axiom from Sundararajan, Tuly, and Yan 2017.

    Axiomatic Attribution for Deep Nets: https://arxiv.org/pdf/1703.01365.pdf

    The attributions must add up to the difference between the anomaly score
    at the input and the baseline.

    In this case we choose one baseline point and one very anomalous point, so
    the difference ought to be nearly 1.
    """
    num_steps = 1000
    interpreter = self._get_test_interpreter(0.95, 250, num_steps)

    # Create an anomalous data point by inducing a large perturbation.
    anomalous_data = np.array(_TEST_ARRAY_4) + np.array(
        [0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0])
    test_anomaly = pd.Series(data=anomalous_data, index=_FIELDS)

    # Get the attribution, where df_grad contains the gradients at each point.
    _, reference_point_dict, df_grad = interpreter.blame(test_anomaly)

    # Convert the nearest reference point, and get models assessment.
    nearest_reference_point = pd.Series(reference_point_dict)
    df_nearest_reference_point = nearest_reference_point.to_frame().T
    x = np.float32(np.matrix(df_nearest_reference_point))
    reference_point_score = interpreter._model.predict(x, verbose=1, steps=1)[0]
    assert reference_point_score == pytest.approx(0.966, abs=1e-3)

    # Get the model's score for the anomaly. Should be very small.
    anomaly_score = interpreter._model.predict(
        np.matrix(anomalous_data), verbose=1, steps=1)[0]

    assert anomaly_score == pytest.approx(0.0, abs=1e-3)

    # Verify that the gradient matrix contains exactly num_steps.
    assert len(df_grad) == num_steps

    # Compute the difference between reference point and the anomaly.
    delta = np.array(nearest_reference_point - test_anomaly)

    # Compute the score difference.
    score_diff = reference_point_score - anomaly_score

    # Compute equation 3 from Sundararajan, Tuly, and Yan 2017.
    cumulative = (np.array(df_grad.cumsum(axis=0).iloc[-1] / float(num_steps)) *
                  delta).sum()

    # Assert the completeness axiom.
    assert score_diff == pytest.approx(cumulative, abs=1e-3)

  def test_completeness_axiom_on_small_perturbation(self):
    """Tests the Completeness Axiom from Sundararajan, Tuly, and Yan 2017.

    Axiomatic Attribution for Deep Nets: https://arxiv.org/pdf/1703.01365.pdf

    The attributions must add up to the difference between the anomaly score
    at the input and the baseline.

    In this case we choose one baseline point and one slightly anomalous point,
    so the difference ought to be nearly 1.
    """
    num_steps = 1000
    interpreter = self._get_test_interpreter(0.95, 250, num_steps)

    # Create an anomalous data point by inducing a large perturbation.
    anomalous_data = np.array(_TEST_ARRAY_4) + np.array(
        [0, 0, 0, 0, 0, 0, 0, 0.05, 0, 0, 0, 0, 0, 0, 0.05, 0])
    test_anomaly = pd.Series(data=anomalous_data, index=_FIELDS)

    # Get the attribution, where df_grad contains the gradients at each point.
    _, reference_point_dict, df_grad = interpreter.blame(test_anomaly)

    # Convert the nearest reference point, and get models assessment.
    nearest_reference_point = pd.Series(reference_point_dict)
    df_nearest_reference_point = nearest_reference_point.to_frame().T
    x = np.float32(np.matrix(df_nearest_reference_point))
    reference_point_score = interpreter._model.predict(x, verbose=1, steps=1)[0]
    assert reference_point_score == pytest.approx(0.967, abs=0.001)

    # Get the model's score for the anomaly. Should be very small.
    anomaly_score = interpreter._model.predict(
        np.matrix(anomalous_data), verbose=1, steps=1)[0]

    assert anomaly_score == pytest.approx(0.169, abs=0.001)

    # Verify that the gradient matrix contains exactly num_steps.
    assert len(df_grad) == num_steps

    # Compute the difference between reference point and the anomaly.
    delta = np.array(nearest_reference_point - test_anomaly)

    # Compute the score difference.
    score_diff = reference_point_score - anomaly_score

    # Compute equation 3 from Sundararajan, Tuly, and Yan 2017.
    cumulative = (np.array(df_grad.cumsum(axis=0).iloc[-1] / float(num_steps)) *
                  delta).sum()

    # Assert the completeness axiom.
    assert score_diff == pytest.approx(cumulative, abs=0.001)

  def _get_test_interpreter(self,
                            min_class_confidence,
                            max_baseline_size,
                            num_steps_integrated_gradients=2000):
    df_pos = pd.read_csv(
        os.path.join(
            os.path.dirname(__file__), _TEST_DATA, _POSITIVE_SAMPLE_FILE),
        index_col=0)
    model_file_path = os.path.join(
        os.path.dirname(__file__), _TEST_DATA, _MODEL_FILENAME)
    model = tf.keras.models.load_model(model_file_path)

    # Load the previous trained and saved model.
    normalization_info = sample_utils.get_normalization_info(df_pos)
    df_pos_normalized = sample_utils.normalize(df_pos, normalization_info)
    interpreter = madi.detectors.integrated_gradients_interpreter.IntegratedGradientsInterpreter(
        model, df_pos_normalized, min_class_confidence, max_baseline_size,
        num_steps_integrated_gradients)
    return interpreter
