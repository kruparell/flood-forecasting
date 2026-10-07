"""Display units for the NSE skill score (SS_NSE) across all da_eval plots and tables.

SS_NSE = (NSE - NSE_base) / (1 - NSE_base). With ``ss_percent`` on (default) it is *displayed* as a percentage
(x 100, i.e. the % reduction in squared forecast error vs the open-loop). Only presentation changes: data,
thresholds, selection, colour limits and function arguments stay in native units (e.g. ``vmin=-0.2`` or
``bin_thresholds="0.01, 0.05"`` still mean -20% / 1% / 5%).

Usage in plotting code (SS axes / colourbars / text only; never on NSE, KGE or ΔNSE axes):
    from da_eval import units
    units.ss_axis(ax, "y")                 # tick labels -> "10%"
    units.ss_colorbar(cbar)                # colourbar ticks -> "%"
    ax.set_ylabel(units.ss_label("NSE skill score"))   # -> "NSE skill score (%)"
    f"{units.fmt_ss(v)}"                   # 0.145 -> "+14.5%"  (or "+0.145" when off)
    df.style.format(units.ss_formatter())  # tables
"""

from typing import Callable, Optional

from matplotlib.ticker import FuncFormatter

_STATE = {"ss_percent": True}


def set_ss_percent(on: bool = True) -> None:
    """Notebook-wide switch: show NSE skill score as % (True) or as a fraction (False)."""
    _STATE["ss_percent"] = bool(on)


def ss_percent() -> bool:
    return _STATE["ss_percent"]


def _trim(s: str) -> str:
    return s.rstrip("0").rstrip(".") if "." in s else s


def fmt_ss(v, decimals: Optional[int] = None, sign: bool = True) -> str:
    """Text for one skill-score value: +14.5% (percent mode, 1 dp) or +0.145 (fraction mode, 3 dp)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    if v != v:  # NaN
        return "–"
    sp = "+" if sign else ""
    if ss_percent():
        d = 1 if decimals is None else decimals
        return f"{v * 100:{sp}.{d}f}%"
    d = 3 if decimals is None else decimals
    return f"{v:{sp}.{d}f}"


def ss_formatter(decimals: Optional[int] = None, sign: bool = True) -> Callable:
    """Callable for pandas Styler.format / apply on SS columns."""
    return lambda v: fmt_ss(v, decimals=decimals, sign=sign)


def ss_tick_formatter() -> FuncFormatter:
    """Matplotlib tick formatter: 0.1 -> '10%' (percent mode) or '0.1' (fraction mode)."""
    if ss_percent():
        return FuncFormatter(lambda v, _pos: (_trim(f"{v * 100:.2f}") + "%").replace("-", "\u2212"))
    return FuncFormatter(lambda v, _pos: _trim(f"{v:.3f}"))


def ss_axis(ax, which: str = "y") -> None:
    """Apply the SS tick formatter to ``ax`` ('x', 'y' or 'both'). No-op in fraction mode."""
    if not ss_percent() or ax is None:
        return
    if which in ("x", "both"):
        ax.xaxis.set_major_formatter(ss_tick_formatter())
    if which in ("y", "both"):
        ax.yaxis.set_major_formatter(ss_tick_formatter())


def ss_colorbar(cbar) -> None:
    """Apply the SS tick formatter to a matplotlib colourbar."""
    if not ss_percent() or cbar is None:
        return
    cbar.formatter = ss_tick_formatter()
    cbar.update_ticks()


def ss_label(text: str) -> str:
    """Append ' (%)' to an SS axis / colourbar / column label in percent mode (idempotent)."""
    if not ss_percent() or "(%)" in text:
        return text
    return f"{text} (%)"


def ss_scale(v):
    """Numeric value for display-only contexts that need numbers (e.g. plotly hover): x100 in percent mode."""
    return v * 100 if ss_percent() else v


def ss_unit() -> str:
    return "%" if ss_percent() else ""
