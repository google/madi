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
"""Tests for google3.third_party.py.madi.datasets.smart_buildings_dataset."""

from absl.testing import absltest
from madi.datasets import smart_buildings_dataset
import pandas as pd
from pandas.util.testing import assert_series_equal


class SmartBuildingsDatasetTest(absltest.TestCase):

  def test_smart_buildings_dataset(self):
    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    ds = smart_buildings_dataset.SmartBuildingsDataset()
    self.assertLen(ds.sample, 60425)
    self.assertAlmostEqual(
        ds.sample['data:zone_air_heating_temperature_setpoint'].mean(),
        290.627693, 4)
    self.assertAlmostEqual(
        ds.sample['data:zone_air_heating_temperature_setpoint'].std(), 3.703542,
        4)

    self.assertAlmostEqual(ds.sample['data:zone_air_temperature_sensor'].mean(),
                           295.475594, 4)
    self.assertAlmostEqual(ds.sample['data:zone_air_temperature_sensor'].std(),
                           0.971860, 4)

    self.assertAlmostEqual(
        ds.sample['data:zone_air_cooling_temperature_setpoint'].mean(),
        299.942589, 4)
    self.assertAlmostEqual(
        ds.sample['data:zone_air_cooling_temperature_setpoint'].std(), 2.773154,
        4)

    self.assertAlmostEqual(ds.sample['data:supply_air_flowrate_sensor'].mean(),
                           0.077608, 4)
    self.assertAlmostEqual(ds.sample['data:supply_air_flowrate_sensor'].std(),
                           0.100314, 4)

    self.assertAlmostEqual(
        ds.sample['data:supply_air_damper_percentage_command'].mean(),
        45.299588, 4)
    self.assertAlmostEqual(
        ds.sample['data:supply_air_damper_percentage_command'].std(), 39.005507,
        4)

    self.assertAlmostEqual(
        ds.sample['data:supply_air_flowrate_setpoint'].mean(), 0.079952, 4)
    self.assertAlmostEqual(ds.sample['data:supply_air_flowrate_setpoint'].std(),
                           0.089611, 4)

    assert_series_equal(
        ds.sample['class_label'].value_counts(),
        pd.Series([58504, 1921], name='class_label', index=[1, 0]))

  def test_readme(self):
    ds = smart_buildings_dataset.SmartBuildingsDataset()
    self.assertLen(ds.description, 1627)

if __name__ == '__main__':
  absltest.main()
