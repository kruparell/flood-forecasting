"""Single, notebook-wide definition of "Global-Best" and "Per-Basin Best" reference models.

A :class:`SelectionPolicy` says how configurations are ranked; :func:`resolve_selection` turns it into a
:class:`Selection` (the chosen Config IDs) once per session. Every plot, table and dashboard reads the
*active* selection (set by ``DAAnalysisSession.set_selection``) instead of choosing its own models.

Per-basin score = ``metric`` aggregated (``agg``) over ``leads``; a basin counts only if every selected lead
is finite.

* Global-Best DA: the DA config with the best ``global_stat`` of that score across basins (on the basins
  common to all well-covered candidates when ``common_basins``).
* Per-Basin DA: in every basin, the DA config with the best score.
* Global-Best / Per-Basin PP: the same two rules applied to the fixed-rho PP post-processors.

Selection is in-sample (chosen on the evaluation period), so Per-Basin models are oracle upper bounds.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from da_eval.ingestion import _ensure_skill_score_columns, is_rho_config

PER_BASIN_DA = "Per-Basin Best DA"
PER_BASIN_RHO = "AR1_Per_Basin_Best_Rho"

GLOBAL_STATS = {
    "median": lambda v: float(np.median(v)),
    "mean": lambda v: float(np.mean(v)),
    "q10": lambda v: float(np.quantile(v, 0.10)),
    "q25": lambda v: float(np.quantile(v, 0.25)),
    "pct_improved": lambda v: float((v > 0).mean() * 100),
}
METRIC_SHORT = {
    "NSE Skill Score": "NSE skill score", "NSE Delta": "ΔNSE", "DA NSE": "NSE",
    "KGE Skill Score": "KGE skill score", "KGE Delta": "ΔKGE", "DA KGE": "KGE",
    "Pearson-r Skill Score": "Pearson-r skill score", "DA Pearson-r": "Pearson-r",
    "Alpha-NSE Skill Score": "Alpha-NSE skill score", "DA Alpha-NSE": "Alpha-NSE",
    "Beta-KGE Skill Score": "Beta-KGE skill score", "DA Beta-KGE": "Beta-KGE",
    "Beta-NSE Skill Score": "Beta-NSE skill score", "DA Beta-NSE": "Beta-NSE",
    "NSE Skill Score (t+0 in-window)": "t+0 in-window NSE skill score",
}


def _lead_str(leads: Sequence[int]) -> str:
    leads = sorted(leads)
    if len(leads) == 1:
        return f"t+{leads[0]}"
    if leads == list(range(leads[0], leads[-1] + 1)):
        return f"t+{leads[0]}–{leads[-1]}"
    return "t+" + ",".join(str(l) for l in leads)


@dataclass(frozen=True)
class SelectionPolicy:
    """How reference models are chosen (see module docstring)."""

    metric: str = "NSE Skill Score"
    leads: Tuple[int, ...] = (1,)
    agg: str = "mean"                 # per basin over leads: "mean" | "sum"
    global_stat: str = "median"       # across basins: median | mean | q10 | q25 | pct_improved
    common_basins: bool = True
    candidates: Union[str, Tuple[str, ...]] = "da"   # "da" (all DA configs), a regex, or explicit Config IDs
    min_coverage: float = 0.9

    def __post_init__(self):
        object.__setattr__(self, "leads", tuple(sorted({int(l) for l in np.atleast_1d(self.leads)})))
        if not isinstance(self.candidates, str):
            object.__setattr__(self, "candidates", tuple(self.candidates))
        if self.global_stat not in GLOBAL_STATS:
            raise ValueError(f"global_stat must be one of {list(GLOBAL_STATS)}")
        if self.agg not in ("mean", "sum"):
            raise ValueError("agg must be 'mean' or 'sum'")

    def label(self) -> str:
        m = METRIC_SHORT.get(self.metric, self.metric)
        if len(self.leads) == 1:
            return f"{self.global_stat} {m} at {_lead_str(self.leads)}"
        return f"{self.global_stat} of {self.agg} {m} over {_lead_str(self.leads)}"


@dataclass(frozen=True)
class Selection:
    policy: SelectionPolicy
    global_da: str
    global_rho: Optional[str]
    per_basin_da: Dict[str, str] = field(repr=False)
    per_basin_rho: Dict[str, str] = field(repr=False)
    ranking: pd.DataFrame = field(repr=False)
    n_basins: int = 0

    def label(self) -> str:
        return f"selected on {self.policy.label()}"

    def summary(self) -> pd.DataFrame:
        return self.ranking


def _is_pool_da(cid: str) -> bool:
    c = str(cid).lower()
    return not (is_rho_config(cid) or "baseline" in c or "per-basin" in c or "per_basin" in c)


def _is_pool_rho(cid: str) -> bool:
    c = str(cid).lower()
    return is_rho_config(cid) and "per_basin" not in c and "per-basin" not in c


def _candidates(cfgs, policy: SelectionPolicy):
    da = [c for c in cfgs if _is_pool_da(c)]
    cand = policy.candidates
    if isinstance(cand, tuple):
        da = [c for c in da if c in set(cand)]
    elif cand not in ("da", "all", ""):
        da = [c for c in da if re.search(cand, str(c))]
    return da, [c for c in cfgs if _is_pool_rho(c)]


def _pool_scores(df: pd.DataFrame, pool, policy: SelectionPolicy) -> pd.DataFrame:
    d = df[df["Config ID"].isin(pool) & df["Lead Time (Days)"].isin(policy.leads)]
    if d.empty:
        return pd.DataFrame(columns=["Config ID", "Basin ID", "score"])
    wide = d.pivot_table(index=["Config ID", "Basin ID"], columns="Lead Time (Days)", values=policy.metric,
                         aggfunc="first").reindex(columns=list(policy.leads))
    wide = wide[wide.notna().all(axis=1)]
    s = wide.sum(axis=1) if policy.agg == "sum" else wide.mean(axis=1)
    return s.rename("score").reset_index()


def _rank(scores: pd.DataFrame, policy: SelectionPolicy, kind: str):
    if scores.empty:
        return None, {}, pd.DataFrame(), 0
    cover = scores.groupby("Config ID")["Basin ID"].nunique()
    kept = cover[cover >= policy.min_coverage * cover.max()].index
    sc = scores[scores["Config ID"].isin(kept)]
    if policy.common_basins:
        n_cfg = sc.groupby("Basin ID")["Config ID"].nunique()
        sc_g = sc[sc["Basin ID"].isin(n_cfg[n_cfg == len(kept)].index)]
    else:
        sc_g = sc
    stat = GLOBAL_STATS[policy.global_stat]
    rank = sc_g.groupby("Config ID")["score"].agg(stat).rename(policy.global_stat)
    rows = pd.DataFrame({policy.global_stat: rank, "N basins": sc_g.groupby("Config ID")["Basin ID"].nunique()})
    rows = rows.sort_values(policy.global_stat, ascending=False)
    best = str(rows.index[0]) if not rows.empty else None
    pb = scores.loc[scores.groupby("Basin ID")["score"].idxmax()]
    per_basin = dict(zip(pb["Basin ID"].astype(str), pb["Config ID"].astype(str)))
    share = pd.Series(list(per_basin.values())).value_counts(normalize=True) * 100
    rows["% basins best"] = share.reindex(rows.index).fillna(0.0)
    rows.insert(0, "Pool", kind)
    rows["Global-Best"] = rows.index == best
    return best, per_basin, rows.reset_index(), int(sc_g["Basin ID"].nunique())


def resolve_selection(df_eval: pd.DataFrame, policy: Optional[SelectionPolicy] = None) -> Selection:
    policy = policy or SelectionPolicy()
    df = _ensure_skill_score_columns(df_eval)
    if policy.metric not in df.columns:
        raise ValueError(f"metric {policy.metric!r} not in evaluation columns")
    cfgs = df["Config ID"].dropna().unique()
    da, rho = _candidates(cfgs, policy)
    if not da:
        raise ValueError("No DA candidate configs match the selection policy")
    gb, pb, rank_da, n = _rank(_pool_scores(df, da, policy), policy, "DA")
    gbr, pbr, rank_rho, _ = _rank(_pool_scores(df, rho, policy), policy, "PP")
    ranking = pd.concat([rank_da, rank_rho], ignore_index=True)
    return Selection(policy, gb, gbr, pb, pbr, ranking, n)


def _synth(df: pd.DataFrame, mapping: Dict[str, str], name: str) -> pd.DataFrame:
    if not mapping:
        return df.iloc[0:0]
    m = pd.DataFrame({"Basin ID": list(mapping), "_sel": list(mapping.values())})
    rows = df[df["Config ID"].isin(set(mapping.values()))].copy()
    rows["_b"] = rows["Basin ID"].astype(str)
    rows = rows.merge(m.rename(columns={"Basin ID": "_b"}), on="_b")
    rows = rows[rows["Config ID"] == rows["_sel"]].drop(columns=["_b", "_sel"])
    rows["Config ID"] = name
    return rows


def apply_selection(df_eval: pd.DataFrame, sel: Selection) -> pd.DataFrame:
    """Rebuilds the synthetic ``Per-Basin Best DA`` / ``AR1_Per_Basin_Best_Rho`` rows from ``sel``."""
    df = df_eval[~df_eval["Config ID"].isin([PER_BASIN_DA, PER_BASIN_RHO])]
    extra = [_synth(df, sel.per_basin_da, PER_BASIN_DA), _synth(df, sel.per_basin_rho, PER_BASIN_RHO)]
    extra = [e for e in extra if not e.empty]
    return pd.concat([df] + extra, ignore_index=True) if extra else df.reset_index(drop=True)


def per_basin_timeseries(df_ts: pd.DataFrame, sel: Selection) -> pd.DataFrame:
    """Adds ``Per-Basin Best DA`` forecast rows to a timeseries frame (replacing any existing ones)."""
    df = df_ts[df_ts["Config ID"] != PER_BASIN_DA]
    pb = _synth(df, sel.per_basin_da, PER_BASIN_DA)
    return pd.concat([df, pb], ignore_index=True) if not pb.empty else df


# ---------------------------------------------------------------------------
# Active selection (set by the session; read by library helpers)
# ---------------------------------------------------------------------------
_ACTIVE: Optional[Selection] = None


def set_active(sel: Optional[Selection]) -> None:
    global _ACTIVE
    _ACTIVE = sel


def get_active() -> Optional[Selection]:
    return _ACTIVE


def policy_from(policy: Optional[SelectionPolicy] = None, **kw) -> SelectionPolicy:
    base = policy or SelectionPolicy()
    return replace(base, **kw) if kw else base
