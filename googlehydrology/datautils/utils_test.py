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

"""Unit tests for googlehydrology.datautils.utils."""

from absl.testing import absltest
from googlehydrology.datautils.utils import (
    check_and_select_basins,
    safe_sel_basins,
    validate_streamflow_series,
)
import numpy as np
import pandas as pd
import xarray as xr


class UtilsTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    dates = pd.date_range('2020-01-01', '2020-01-10')
    basins = ['camels_01013500', 'camels_01022500', 'camels_01030500']
    data = np.ones((len(dates), len(basins)), dtype=np.float32)
    self.ds = xr.Dataset(
        data_vars={
            'streamflow': (['date', 'basin'], data),
            'temp': (['date', 'basin'], data * 2.0),
        },
        coords={
            'date': dates,
            'basin': basins,
        },
    )

  def test_check_and_select_basins_full_match_preserves_order(self):
    requested = ['camels_01030500', 'camels_01013500']
    sliced, missing = check_and_select_basins(self.ds, requested, strict=True)
    self.assertEmpty(missing)
    self.assertEqual(list(sliced.coords['basin'].values), requested)

  def test_check_and_select_basins_partial_match_warning(self):
    requested = ['camels_01013500', 'missing_basin_1', 'missing_basin_2']
    sliced, missing = check_and_select_basins(self.ds, requested, strict=False)
    self.assertEqual(missing, ['missing_basin_1', 'missing_basin_2'])
    self.assertEqual(list(sliced.coords['basin'].values), ['camels_01013500'])

  def test_check_and_select_basins_strict_raises_keyerror(self):
    requested = ['camels_01013500', 'missing_basin_1']
    with self.assertRaises(KeyError) as ctx:
      check_and_select_basins(self.ds, requested, strict=True)
    self.assertIn('missing_basin_1', str(ctx.exception))

  def test_check_and_select_basins_case_insensitive(self):
    requested = ['CAMELS_01013500', 'camels_01022500']
    sliced, missing = check_and_select_basins(self.ds, requested, strict=True)
    self.assertEmpty(missing)
    self.assertEqual(
        list(sliced.coords['basin'].values),
        ['camels_01013500', 'camels_01022500'],
    )

  def test_check_and_select_basins_zero_match_raises_keyerror(self):
    requested = ['unknown_1', 'unknown_2']
    with self.assertRaises(KeyError):
      check_and_select_basins(self.ds, requested, strict=False)

  def test_safe_sel_basins_wrapper(self):
    requested = ['camels_01013500', 'missing_basin']
    sliced = safe_sel_basins(self.ds, requested)
    self.assertEqual(list(sliced.coords['basin'].values), ['camels_01013500'])

  def test_validate_streamflow_series_valid(self):
    # Valid complete streamflow passes without error
    validate_streamflow_series(
        self.ds.sel(basin='camels_01013500'),
        basin_id='camels_01013500',
        start_date='2020-01-01',
        end_date='2020-01-05',
        strict=True,
    )

  def test_validate_streamflow_series_nan_strict_raises(self):
    # Inject NaN into streamflow
    ds_nan = self.ds.sel(basin='camels_01013500').copy(deep=True)
    ds_nan['streamflow'].loc[dict(date='2020-01-03')] = np.nan

    with self.assertRaises(ValueError) as ctx:
      validate_streamflow_series(
          ds_nan,
          basin_id='camels_01013500',
          start_date='2020-01-01',
          end_date='2020-01-05',
          strict=True,
      )
    self.assertIn('contains 1 NaNs', str(ctx.exception))
    self.assertIn('2020-01-03', str(ctx.exception))

  def test_validate_streamflow_series_empty_strict_raises(self):
    # Slicing out of date range produces empty records
    with self.assertRaises(ValueError) as ctx:
      validate_streamflow_series(
          self.ds.sel(basin='camels_01013500'),
          basin_id='camels_01013500',
          start_date='2025-01-01',
          end_date='2025-01-05',
          strict=True,
      )
    self.assertIn('completely EMPTY', str(ctx.exception))


if __name__ == '__main__':
  absltest.main()
