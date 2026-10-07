"""Plotting and visualization utilities for hydrological model runs."""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_hydrograph(
    obs: np.ndarray,
    sim: np.ndarray,
    dates: Optional[List[str]] = None,
    title: str = "Streamflow Hydrograph",
    save_path: Optional[Union[str, Path]] = None,
    figsize: tuple = (12, 5)
):
    """Plots observed vs simulated hydrograph."""
    fig, ax = plt.subplots(figsize=figsize)
    x = dates if dates is not None else np.arange(len(obs))

    ax.plot(x, obs, label="Observed Streamflow", color="black", linewidth=1.5)
    ax.plot(x, sim, label="Simulated Streamflow", color="blue", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Date / Timestep")
    ax.set_ylabel("Streamflow (m³/s)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    return fig, ax
