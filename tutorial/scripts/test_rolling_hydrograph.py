"""Unit tests for rolling hydrograph extraction and visualization."""

import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

_SCRIPT_DIR = Path(__file__).resolve().parent
_TUTORIAL_DIR = _SCRIPT_DIR.parent
_REPO_ROOT = _TUTORIAL_DIR.parent

for _p in [_REPO_ROOT, _TUTORIAL_DIR, _SCRIPT_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import backend


def test_calculate_timeseries_metrics():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sim = np.array([1.1, 1.9, 3.2, 3.9, 5.1])
    m = backend.calculate_timeseries_metrics(obs, sim)
    assert 'NSE' in m and 'KGE' in m and 'Pearson-r' in m and 'RMSE' in m
    assert m['NSE'] > 0.95
    assert m['Pearson-r'] > 0.95
    assert m['RMSE'] < 0.2


def test_extract_rolling_hydrograph_synthetic_dataset():
    dates = pd.date_range('2011-01-01', periods=100, freq='D')
    time_steps = np.array([-1, 0, 1, 2, 3, 4, 5, 6, 7])
    basins = ['camels_01054200']

    sim_data = np.random.rand(1, 1, 100, len(time_steps)).astype(np.float32)
    obs_data = np.random.rand(1, 1, 100, len(time_steps)).astype(np.float32)

    ds = xr.Dataset(
        data_vars={
            'streamflow_sim': (('basin', 'freq', 'date', 'time_step'), sim_data),
            'streamflow_obs': (('basin', 'freq', 'date', 'time_step'), obs_data),
        },
        coords={'basin': basins, 'freq': ['1D'], 'date': dates, 'time_step': time_steps}
    )

    lead_time = 5
    df = backend.extract_rolling_hydrograph(ds, basin_id='camels_01054200', lead_time=lead_time, window_days=50)
    assert len(df) == 50
    assert df.index[0] == dates[0] + pd.Timedelta(days=lead_time)
    assert 'sim' in df.columns and 'obs' in df.columns


def test_extract_rolling_hydrograph_dict():
    n_days = 365
    obs = np.linspace(1.0, 10.0, n_days)
    sim = obs + 0.1
    data_dict = {'camels_01054200_obs': obs, 'camels_01054200_sim': sim}

    df = backend.extract_rolling_hydrograph(data_dict, basin_id='camels_01054200', lead_time=5, window_days=365)
    assert len(df) == 365
    assert 'sim' in df.columns and 'obs' in df.columns


def test_plot_rolling_hydrograph():
    dates = pd.date_range('2011-01-01', periods=100, freq='D')
    obs = np.linspace(1, 5, 100)
    sim = obs + 0.1
    df = pd.DataFrame({'sim': sim, 'obs': obs}, index=dates)

    fig, ax = backend.plot_rolling_hydrograph(
        data=df,
        basin_id='camels_01054200',
        lead_time=5,
        window_days=100
    )
    assert fig is not None and ax is not None
    assert "camels_01054200" in ax.get_title()
    assert "Lead Time" in ax.get_title() and "5" in ax.get_title()
    plt.close(fig)


if __name__ == '__main__':
    pytest.main([__file__])
