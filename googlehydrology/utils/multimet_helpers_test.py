# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for googlehydrology.utils.multimet_helpers."""

from absl.testing import absltest
from googlehydrology.utils.multimet_helpers import (
    _get_source_var,
    norm_var,
    prepare_multimet_batch,
)
import numpy as np
import pandas as pd
import xarray as xr


class DummyConfig:

  def __init__(self):
    self.hindcast_inputs = {'group1': ['hres_total_precipitation']}
    self.forecast_inputs = {'group1': ['hres_total_precipitation']}
    self.static_attributes = ['area']
    self.hidden_size = 8
    self.union_mapping = {}


class MultiMetHelpersTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    # Mock scaler dataset with canonical ERA5-Land parameters
    param_coord = ['mean', 'std', 'center', 'scale']
    self.scaler = xr.Dataset(
        data_vars={
            'era5land_surface_net_solar_radiation': (
                ['parameter'],
                np.array([150.0, 50.0, 150.0, 50.0], dtype=np.float32),
            ),
            'era5land_total_precipitation': (
                ['parameter'],
                np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            ),
            'area': (
                ['parameter'],
                np.array([100.0, 10.0, 100.0, 10.0], dtype=np.float32),
            ),
            'streamflow': (
                ['parameter'],
                np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
            ),
        },
        coords={
            'parameter': param_coord,
        },
    )

  def test_get_source_var_canonical_mappings(self):
    union_map = {'custom_fc': 'era5land_temperature_2m'}
    self.assertEqual(
        _get_source_var(union_map, 'custom_fc'), 'era5land_temperature_2m'
    )
    self.assertEqual(
        _get_source_var(union_map, 'hres_surface_net_solar_radiation'),
        'era5land_surface_net_solar_radiation',
    )
    self.assertEqual(
        _get_source_var(union_map, 'hres_total_precipitation'),
        'era5land_total_precipitation',
    )
    self.assertEqual(
        _get_source_var(union_map, 'cpc_total_precipitation'),
        'era5land_total_precipitation',
    )
    self.assertEqual(
        _get_source_var(union_map, 'imerg_precipitation'),
        'era5land_total_precipitation',
    )
    self.assertEqual(
        _get_source_var(union_map, 'graphcast_total_precipitation'),
        'era5land_total_precipitation',
    )

  def test_norm_var_source_mapping_successful(self):
    # Test normalizing HRES forecast radiation using canonical ERA5-Land statistics
    raw_vals = np.array([100.0, 150.0, 200.0], dtype=np.float32)
    normed = norm_var(
        self.scaler,
        var_name='hres_surface_net_solar_radiation',
        val=raw_vals,
        source_var='era5land_surface_net_solar_radiation',
    )
    expected = (raw_vals - 150.0) / 50.0
    np.testing.assert_allclose(normed, expected, rtol=1e-5)

  def test_norm_var_missing_raises_keyerror(self):
    raw_vals = np.array([10.0, 20.0], dtype=np.float32)
    with self.assertRaises(KeyError) as ctx:
      norm_var(
          self.scaler,
          var_name='completely_unknown_feature',
          val=raw_vals,
          source_var='also_unknown',
      )
    self.assertIn('completely_unknown_feature', str(ctx.exception))

  def test_norm_var_scalar_float(self):
    val = 200.0
    normed = norm_var(
        self.scaler,
        var_name='era5land_surface_net_solar_radiation',
        val=val,
    )
    self.assertAlmostEqual(normed, 1.0, places=5)

  def test_prepare_multimet_batch_date_alignment_no_duplicate_forcing(self):
    # Setup synthetic dates: issue_date t0 = '2020-01-10'
    # hindcast_window_days = 5 -> dates_h: '2020-01-06' .. '2020-01-10'
    # forecast_lead_days = 3 -> dates_f: '2020-01-06' .. '2020-01-13' (8 steps)
    # Construct synthetic MultiMet dataset where value = 100 * issue_day + lead_time_days
    # Since valid_day = issue_day + lead_time_days, any forecast valid for day D
    # issued on day (D - lead_time_days) has value: 100 * (D - L) + L.
    all_dates = pd.date_range('2020-01-01', '2020-01-20').strftime('%Y-%m-%d').values
    lead_times = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)

    data = np.zeros((len(all_dates), len(lead_times)), dtype=np.float32)
    for i, d_str in enumerate(all_dates):
      day_num = int(d_str.split('-')[-1])
      for j, l_val in enumerate(lead_times):
        # Encode valid day explicitly in value: valid_day = day_num + l_val
        data[i, j] = float(day_num + l_val)

    multimet_ds = xr.Dataset(
        data_vars={
            'hres_total_precipitation': (['date', 'lead_time'], data),
        },
        coords={
            'date': all_dates,
            'lead_time': lead_times,
        },
    )

    ds_caravan = xr.Dataset(
        data_vars={
            'streamflow': (['date'], np.ones(len(all_dates), dtype=np.float32)),
            'total_precipitation_sum': (
                ['date'],
                np.ones(len(all_dates), dtype=np.float32),
            ),
            'temperature_2m_mean': (
                ['date'],
                np.ones(len(all_dates), dtype=np.float32),
            ),
            'surface_net_solar_radiation_mean': (
                ['date'],
                np.ones(len(all_dates), dtype=np.float32),
            ),
            'surface_net_thermal_radiation_mean': (
                ['date'],
                np.ones(len(all_dates), dtype=np.float32),
            ),
            'surface_pressure_mean': (
                ['date'],
                np.ones(len(all_dates), dtype=np.float32),
            ),
        },
        coords={'date': all_dates},
    )
    caravan_attrs = pd.DataFrame({'area': [100.0]}, index=['basin_test'])

    cfg = DummyConfig()
    batch = prepare_multimet_batch(
        mode='multimet_0_and_1_to_7',
        basin_id='basin_test',
        issue_date='2020-01-10',
        cfg=cfg,
        scaler=self.scaler,
        caravan_attrs=caravan_attrs,
        ds_caravan=ds_caravan,
        hindcast_window_days=5,
        forecast_lead_days=3,
        multimet_ds=multimet_ds,
    )

    # Expected valid days for the 8 steps ('2020-01-06' to '2020-01-13'):
    # [6, 7, 8, 9, 10, 11, 12, 13]
    expected_valid_days = np.array([6, 7, 8, 9, 10, 11, 12, 13], dtype=np.float32)
    f_vals = batch['x_d_forecast']['hres_total_precipitation'][0, :, 0].numpy()
    np.testing.assert_allclose(f_vals, expected_valid_days, rtol=1e-5)

    # Specifically verify no duplicate weather forcing at t0 (index 4, day 10)
    # vs t0+1 (index 5, day 11)
    self.assertNotEqual(f_vals[4], f_vals[5])
    self.assertEqual(f_vals[4], 10.0)
    self.assertEqual(f_vals[5], 11.0)

    # Verify hindcast_dict has valid days [6, 7, 8, 9, 10] then 3 NaNs
    h_vals = batch['x_d']['hres_total_precipitation'][0, :, 0].numpy()
    np.testing.assert_allclose(h_vals[:5], np.array([6, 7, 8, 9, 10], dtype=np.float32))
    self.assertTrue(np.isnan(h_vals[5:]).all())

    # Test reanalysis_and_fixed_leadtime mode with fixed_leadtime=2 (3-day lead)
    batch_fixed = prepare_multimet_batch(
        mode='reanalysis_and_fixed_leadtime',
        basin_id='basin_test',
        issue_date='2020-01-10',
        cfg=cfg,
        scaler=self.scaler,
        caravan_attrs=caravan_attrs,
        ds_caravan=ds_caravan,
        hindcast_window_days=5,
        forecast_lead_days=3,
        fixed_leadtime=2,
        multimet_ds=multimet_ds,
    )
    f_vals_fixed = batch_fixed['x_d_forecast']['hres_total_precipitation'][
        0, :, 0
    ].numpy()
    np.testing.assert_allclose(f_vals_fixed, expected_valid_days, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()
