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
"""Tests for madi.datasets.smart_buildings_dataset."""

from madi.datasets import smart_buildings_dataset
import pandas as pd
from pandas.util.testing import assert_series_equal
import pytest


class TestSmartBuildingsDataset:

  def test_smart_buildings_dataset(self):
    pd.set_option('display.max_rows', None, 'display.max_columns', None)
    ds = smart_buildings_dataset.SmartBuildingsDataset()
    assert len(ds.sample) == 60425
    assert ds.sample['data:zone_air_heating_temperature_setpoint'].mean(
    ) == pytest.approx(
        290.627693, abs=1e-4)
    assert ds.sample['data:zone_air_heating_temperature_setpoint'].std(
    ) == pytest.approx(
        3.703542, abs=1e-4)

    assert ds.sample['data:zone_air_temperature_sensor'].mean(
    ) == pytest.approx(
        295.475594, abs=1e-4)
    assert ds.sample['data:zone_air_temperature_sensor'].std() == pytest.approx(
        0.971860, abs=1e-4)

    assert ds.sample['data:zone_air_cooling_temperature_setpoint'].mean(
    ) == pytest.approx(
        299.942589, abs=1e-4)
    assert ds.sample['data:zone_air_cooling_temperature_setpoint'].std(
    ) == pytest.approx(
        2.773154, abs=1e-4)

    assert ds.sample['data:supply_air_flowrate_sensor'].mean() == pytest.approx(
        0.077608, abs=1e-4)
    assert ds.sample['data:supply_air_flowrate_sensor'].std() == pytest.approx(
        0.100314, abs=1e-4)

    assert ds.sample['data:supply_air_damper_percentage_command'].mean(
    ) == pytest.approx(
        45.299588, abs=1e-4)
    assert ds.sample['data:supply_air_damper_percentage_command'].std(
    ) == pytest.approx(
        39.005507, abs=1e-4)

    assert ds.sample['data:supply_air_flowrate_setpoint'].mean(
    ) == pytest.approx(
        0.079952, abs=1e-4)
    assert ds.sample['data:supply_air_flowrate_setpoint'].std(
    ) == pytest.approx(
        0.089611, abs=1e-4)

    assert_series_equal(
        ds.sample['class_label'].value_counts(),
        pd.Series([58504, 1921], name='class_label', index=[1, 0]))

  def test_readme(self):
    ds = smart_buildings_dataset.SmartBuildingsDataset()
    assert len(ds.description) == 1627
