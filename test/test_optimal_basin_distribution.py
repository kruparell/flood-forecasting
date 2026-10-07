"""Unit test suite for Top-N Optimal Basin Distribution table and visualizations."""

import sys
from pathlib import Path
import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = Path("/usr/local/google/home/kruparell/openhydronets_next")
sys.path.insert(0, str(REPO_ROOT / "tutorial/utils"))

from da_eval.tables import (
    build_top_n_optimal_basin_distribution_table,
    style_top_n_optimal_basin_distribution_table,
)
from da_eval.static_plots import plot_top_n_optimal_basin_distribution


@pytest.fixture
def synthetic_eval_data():
    np.random.seed(42)
    basins = [f"basin_{i:03d}" for i in range(100)]
    configs = [
        "cfg_w14_lr0.1_ep20_bg0.01_c_fc",
        "cfg_w7_lr0.05_ep50_bg0.001_c_fc",
        "cfg_w30_lr0.2_ep10_bg0.01_c_fc",
        "cfg_w14_lr0.5_ep20_bg0.01_c_fc",
        "cfg_w7_lr0.1_ep20_bg0.01_c_fc",
    ]
    records = []
    for b_idx, b in enumerate(basins):
        for c_idx, c in enumerate(configs):
            # Give config 0 a strong baseline advantage, but config 1 wins specific basins
            if b_idx < 40 and c_idx == 0:
                delta = 0.15 + np.random.uniform(0, 0.05)
            elif 40 <= b_idx < 70 and c_idx == 1:
                delta = 0.18 + np.random.uniform(0, 0.05)
            elif 70 <= b_idx < 90 and c_idx == 2:
                delta = 0.12 + np.random.uniform(0, 0.05)
            else:
                delta = 0.05 + np.random.uniform(-0.05, 0.05)

            records.append({
                "Basin ID": b,
                "Config ID": c,
                "Lead Time (Days)": 1,
                "DA NSE": 0.65 + delta,
                "Base NSE": 0.65,
                "NSE Delta": delta,
            })

    df = pd.DataFrame(records)
    return df


def test_build_optimal_basin_table(synthetic_eval_data):
    table = build_top_n_optimal_basin_distribution_table(synthetic_eval_data, top_n=3, lead_time=1)
    assert not table.empty
    assert len(table) == 3
    assert "Basins Won" in table.columns
    assert "Win Share (%)" in table.columns
    assert "Is Global Best" in table.columns
    assert table["Basins Won"].sum() <= 100
    assert table["Basins Won"].iloc[0] >= table["Basins Won"].iloc[1]

    styler = style_top_n_optimal_basin_distribution_table(table)
    assert styler is not None


def test_plot_optimal_basin_distribution(synthetic_eval_data):
    fig_comb = plot_top_n_optimal_basin_distribution(synthetic_eval_data, top_n=4, plot_type="combined")
    assert isinstance(fig_comb, plt.Figure)
    plt.close(fig_comb)

    fig_bar = plot_top_n_optimal_basin_distribution(synthetic_eval_data, top_n=4, plot_type="bar")
    assert isinstance(fig_bar, plt.Figure)
    plt.close(fig_bar)

    fig_pie = plot_top_n_optimal_basin_distribution(synthetic_eval_data, top_n=4, plot_type="donut")
    assert isinstance(fig_pie, plt.Figure)
    plt.close(fig_pie)
