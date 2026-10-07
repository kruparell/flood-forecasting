"""Unit test for ZarrDatasetReader and safe basin selection."""

import unittest
from googlehydrology.datasetzoo.zarr_dataset_reader import safe_sel_basins
import numpy as np
import pandas as pd
import xarray as xr


class ZarrDatasetReaderTest(unittest.TestCase):

  def setUp(self):
    basins = ['US_01013500', 'US_01022500', 'US_01031500']
    dates = pd.date_range('2020-01-01', '2020-01-10')
    data = np.random.rand(len(basins), len(dates))

    self.ds = xr.Dataset(
        data_vars={'streamflow': (['basin', 'date'], data)},
        coords={'basin': basins, 'date': dates},
    )

  def test_safe_sel_basins_exact_match(self):
    sub_ds = safe_sel_basins(self.ds, ['US_01013500', 'US_01022500'])
    self.assertEqual(
        list(sub_ds.coords['basin'].values), ['US_01013500', 'US_01022500']
    )

  def test_safe_sel_basins_case_insensitive_fallback(self):
    sub_ds = safe_sel_basins(self.ds, ['us_01013500'])
    self.assertEqual(list(sub_ds.coords['basin'].values), ['US_01013500'])

  def test_safe_sel_basins_non_existent_basin_returns_empty(self):
    sub_ds = safe_sel_basins(self.ds, ['NON_EXISTENT_BASIN'])
    self.assertEqual(len(sub_ds.coords['basin']), 0)

  def test_safe_sel_basins_missing_coordinate_raises_key_error(self):
    ds_no_coord = xr.Dataset(data_vars={'streamflow': (['x'], [1.0, 2.0])})
    with self.assertRaises(KeyError):
      safe_sel_basins(ds_no_coord, ['US_01013500'])


if __name__ == '__main__':
  unittest.main()





