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
"""Tests for google3.third_party.py.madi.utils.evaluation_utils."""


from absl.testing import absltest
from madi.utils import evaluation_utils


class EvaluationUtilsTest(absltest.TestCase):

  def test_compute_auc_max(self):

    y_actual = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    auc = evaluation_utils.compute_auc(y_actual, y_actual)
    self.assertEqual(auc, 1.0)

  def test_compute_auc_min(self):
    y_actual = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    y_predicted = [float(not bool(y)) for y in y_actual]
    auc = evaluation_utils.compute_auc(y_actual, y_predicted)
    self.assertEqual(auc, 0.0)


if __name__ == '__main__':
  absltest.main()
