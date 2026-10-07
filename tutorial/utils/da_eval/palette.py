"""Notebook-wide colours / line styles for the reference models, and markers for lead times.

Set once (``session.set_palette("cvd_safe")`` in the notebook's settings cell); every da_eval plot reads the
active palette at draw time, so a model has the same colour and line style everywhere.

Encoding used across the notebook:
- colour + line style = model (redundant, so curves stay distinguishable without colour or in greyscale)
- marker = lead time (``lead_marker``), where a plot shows several leads

The default ``"cvd_safe"`` palette (black + IBM/Tol/Okabe-Ito hues) keeps every pair of model colours at
CIELAB dE >= 37 under simulated protanopia, deuteranopia and tritanopia (Machado et al. 2009). ``"legacy"``
restores the earlier Google colours (weakest pair dE ~ 20, e.g. Global vs Per-Basin DA blues).

Usage in plotting code:
    from da_eval import palette
    ax.plot(x, y, color=palette.color("global_da"), ls=palette.ls("global_da"), label=palette.label("global_da"))
    palette.style("per_basin_rho")        # dict(color=..., ls=..., label=...)
    palette.lead_marker(3)                # "s"
    palette.by_label()["Global-Best DA"]  # live {label: colour} mapping for label-keyed code
"""

from collections.abc import Mapping
from typing import Dict

MODEL_KEYS = ("baseline", "global_da", "per_basin_da", "global_rho", "per_basin_rho", "persistence")

LABELS = {
    "baseline": "Baseline",
    "global_da": "Global-Best DA",
    "per_basin_da": "Per-Basin DA",
    "global_rho": "Global-Best PP",
    "per_basin_rho": "Per-Basin PP",
    "persistence": "Persistence",
}

PALETTES: Dict[str, Dict[str, str]] = {
    # DA = cool (violet / cyan), PP post-processing = warm (orange / magenta); Global = first, Per-Basin = second.
    "cvd_safe": {
        "baseline": "#000000",
        "global_da": "#785EF0",
        "per_basin_da": "#33BBEE",
        "global_rho": "#E69F00",
        "per_basin_rho": "#EE3377",
        "persistence": "#999999",
    },
    "legacy": {
        "baseline": "#202124",
        "global_da": "#1a73e8",
        "per_basin_da": "#0d47a1",
        "global_rho": "#e8710a",
        "per_basin_rho": "#a50e0e",
        "persistence": "#9aa0a6",
    },
}

# Global = solid / dash-dot, Per-Basin = dashed / dotted: the line style alone identifies the model.
LINESTYLES = {
    "baseline": "-",
    "global_da": "-",
    "per_basin_da": (0, (5, 2)),
    "global_rho": (0, (6, 2, 1.5, 2)),
    "per_basin_rho": (0, (1.2, 1.8)),
    "persistence": (0, (3, 1, 1, 1, 1, 1)),
}

# Lead-time markers (keyed on the lead number, so a lead keeps its marker when leads are subset).
LEAD_MARKERS = {1: "o", 2: "^", 3: "s", 4: "D", 5: "v", 6: "P", 7: "X"}

_STATE = {"name": "cvd_safe"}


def set_palette(name: str = "cvd_safe") -> None:
    """Notebook-wide switch: ``"cvd_safe"`` (default) or ``"legacy"``."""
    if name not in PALETTES:
        raise ValueError(f"Unknown palette {name!r}; choose from {list(PALETTES)}")
    _STATE["name"] = name


def active() -> str:
    return _STATE["name"]


def color(key: str) -> str:
    return PALETTES[_STATE["name"]][key]


def ls(key: str):
    return LINESTYLES[key]


def label(key: str) -> str:
    return LABELS[key]


def style(key: str) -> dict:
    return dict(color=color(key), ls=ls(key), label=label(key))


def lead_marker(lead: int) -> str:
    """Marker of a lead time; leads beyond t+7 cycle."""
    if lead in LEAD_MARKERS:
        return LEAD_MARKERS[lead]
    keys = sorted(LEAD_MARKERS)
    return LEAD_MARKERS[keys[(int(lead) - 1) % len(keys)]]


_ALIASES = {
    "open-loop baseline": "baseline", "baseline open-loop": "baseline", "baseline": "baseline",
    "global-best da": "global_da", "global best da": "global_da", "global da": "global_da",
    "per-basin da": "per_basin_da", "per-basin best da": "per_basin_da", "per-basin da (oracle)": "per_basin_da",
    "global-best ar(1)": "global_rho", "global ar(1)": "global_rho",
    "per-basin ar(1)": "per_basin_rho",
    "global-best pp": "global_rho", "global pp": "global_rho", "per-basin pp": "per_basin_rho",
    "persistence": "persistence", "naive persistence": "persistence",
}


def key_for(name: str):
    """Model key for a display label / config-like name (None when not a reference model)."""
    if name in LABELS:
        return name
    return _ALIASES.get(str(name).strip().lower())


# Colour-blind-safe qualitative colours for non-model categories (Paul Tol "muted"; avoids the model hues).
QUALITATIVE = ("#332288", "#44AA99", "#DDCC77", "#CC6677", "#88CCEE", "#117733", "#882255", "#999933", "#AA4499")


def value_colors(values) -> dict:
    """{value: colour} for a secondary encoding (e.g. learning rate in risk-return plots).

    Numeric values -> ordered viridis (perceptually uniform, colour-blind safe); other values -> ``QUALITATIVE``.
    ``"legacy"`` palette keeps the earlier tab10 / viridis behaviour.
    """
    import numbers
    import matplotlib.pyplot as plt
    vals = list(values)
    n = len(vals)
    if _STATE["name"] == "legacy":
        cmap = plt.get_cmap("viridis" if n > 8 else "tab10")
        return {v: cmap(i / max(1, n - 1)) if n > 8 else cmap(i) for i, v in enumerate(vals)}
    if all(isinstance(v, numbers.Number) for v in vals) or n > len(QUALITATIVE):
        cmap = plt.get_cmap("viridis")
        return {v: cmap(0.9 * i / max(1, n - 1)) for i, v in enumerate(vals)}
    return {v: QUALITATIVE[i] for i, v in enumerate(vals)}


class _LiveByLabel(Mapping):
    """{label: colour} view that always reflects the active palette."""

    def __getitem__(self, k):
        key = key_for(k)
        if key is None:
            raise KeyError(k)
        return color(key)

    def __iter__(self):
        return iter(LABELS[k] for k in MODEL_KEYS)

    def __len__(self):
        return len(MODEL_KEYS)


def by_label() -> Mapping:
    return _LiveByLabel()
