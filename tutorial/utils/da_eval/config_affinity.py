"""Stage 6b: Configuration-Catchment Affinity.

Asks whether a basin's *static attributes* carry information about *which DA
configuration* performs best on it.

Because per-basin winners are mostly near-ties (median winner - runner-up gap is
~0.01 SS on Stage 45), the analysis is built on the continuous **regret vector**
``r_c = SS_best - SS_c`` rather than the hard argmax label alone, and the
deciding metric is **policy value**: how much of the Global-Best -> Oracle gap
an attribute-driven config choice recovers on basins held out in *spatially
grouped* cross-validation. A shuffled-attribute control guards against leakage.

All reads are read-only (``attributes.zarr``, the pretrained ``config.yml`` and
``scaler.nc``); nothing is written anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import ListedColormap

from da_eval import units
from da_eval.ingestion import _ensure_skill_score_columns

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PRETRAINED_DIR = _REPO_ROOT / "pretrained-models" / "google-floodhub-settings-55-epochs"
DEFAULT_ATTR_ZARR_CANDIDATES = [
    Path("/usr/local/google/home/kruparell/Caravans_V2/attributes.zarr"),
    _REPO_ROOT / "tutorial" / "Caravan-zarr" / "attributes.zarr",
]
COUNTRY_SHP_PATHS = [
    "/usr/local/google/home/kruparell/flood-forecasting/data/countries/ne_110m_admin_0_countries.shp",
    "/usr/local/google/home/kruparell/Caravans_V2/ne_110m_admin_0_countries.shp",
]
SYNTHETIC_CONFIG_PATTERNS = ("per-basin", "oracle", "global best", "tuned per basin", "baseline")
# Non-DA post-processing variants (e.g. PP error correction) are excluded by default:
# the question is which *DA configuration* suits a basin.
DEFAULT_EXCLUDE_CONFIG_PATTERNS = ("postprocess",)

# Colour-blind-friendly qualitative palette (Tableau 10 + extras) for configs / clusters.
QUAL_COLORS = [
    "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc948",
    "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac", "#1b9e77", "#7570b3",
]


# =============================================================================
# 1. Feature space
# =============================================================================
def load_model_static_attribute_names(config_yml: Optional[str | Path] = None) -> List[str]:
    """Reads the ``static_attributes`` list the foundation model was trained on (read-only)."""
    path = Path(config_yml) if config_yml else DEFAULT_PRETRAINED_DIR / "config.yml"
    text = path.read_text()
    try:
        import yaml  # pylint: disable=g-import-not-at-top

        return [str(a) for a in yaml.safe_load(text)["static_attributes"]]
    except Exception:  # Fallback: minimal block parser.
        attrs, in_block = [], False
        for line in text.splitlines():
            if line.startswith("static_attributes:"):
                in_block = True
                continue
            if in_block:
                m = re.match(r"^- (\S+)", line)
                if not m:
                    break
                attrs.append(m.group(1))
        return attrs


def _resolve_attr_zarr(attr_zarr_path: Optional[str | Path]) -> Path:
    for c in [attr_zarr_path, *DEFAULT_ATTR_ZARR_CANDIDATES]:
        if c and Path(c).is_dir():
            return Path(c)
    raise FileNotFoundError("attributes.zarr could not be located.")


def build_static_feature_matrix(
    basins: Sequence[str],
    attr_zarr_path: Optional[str | Path] = None,
    attributes: Optional[Sequence[str]] = None,
    log_skew_threshold: float = 2.0,
    max_missing_frac: float = 0.2,
    use_model_scaler: bool = False,
    scaler_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """Builds a standardized static-attribute matrix for ``basins``.

    Args:
      basins: Basin IDs to include (safely intersected with the zarr index).
      attr_zarr_path: Caravans ``attributes.zarr``; defaults to known local copies.
      attributes: Attribute names; defaults to the model's 84 ``static_attributes``.
      log_skew_threshold: Non-negative attributes with skewness above this get ``log1p``.
      max_missing_frac: Attributes with a larger missing fraction are dropped.
      use_model_scaler: Standardize with the training ``scaler.nc`` centre/scale
        (no log transform) so the space matches what the model saw.
      scaler_path: Override for ``scaler.nc``.

    Returns:
      Dict with ``X`` (standardized DataFrame, index=basin), ``raw`` (untransformed
      values), ``meta`` (lat/lon/country/wmo_reg), ``logged``, ``dropped``.
    """
    attrs = list(attributes) if attributes else load_model_static_attribute_names()
    ds = xr.open_zarr(_resolve_attr_zarr(attr_zarr_path), consolidated=True)
    avail = pd.Index([str(b) for b in ds["basin"].values])
    req = [str(b) for b in basins]
    valid = [b for b in req if b in set(avail)]
    if not valid:
        lower = {b.lower(): b for b in avail}
        valid = [lower[b.lower()] for b in req if b.lower() in lower]
    if not valid:
        raise ValueError("None of the requested basins exist in attributes.zarr.")
    ds = ds.sel(basin=valid)

    present = [a for a in attrs if a in ds]
    dropped = {a: "absent from zarr" for a in attrs if a not in ds}
    raw = pd.DataFrame({a: np.asarray(ds[a].values, dtype=float) for a in present}, index=pd.Index(valid, name="Basin ID"))
    raw = raw.replace([np.inf, -np.inf], np.nan)

    meta = pd.DataFrame(index=raw.index)
    for src, dst in [("gauge_lat", "lat"), ("gauge_lon", "lon")]:
        meta[dst] = np.asarray(ds[src].values, dtype=float) if src in ds else np.nan
    for col in ("country", "wmo_reg", "gauge_name"):
        meta[col] = [str(x) for x in ds[col].values] if col in ds else "Unknown"

    miss = raw.isna().mean()
    for a in miss[miss > max_missing_frac].index:
        dropped[a] = f"{miss[a]:.0%} missing"
    X = raw.drop(columns=[a for a in raw.columns if a in dropped]).copy()

    logged: List[str] = []
    if use_model_scaler:
        sc = xr.open_dataset(Path(scaler_path) if scaler_path else DEFAULT_PRETRAINED_DIR / "scaler.nc")
        for a in list(X.columns):
            if a in sc:
                params = {str(p): float(v) for p, v in zip(sc["parameter"].values, sc[a].values)}
                scale = params.get("scale", np.nan)
                if np.isfinite(scale) and scale > 0:
                    X[a] = (X[a] - params.get("center", 0.0)) / scale
                    continue
            X[a] = (X[a] - X[a].mean()) / (X[a].std(ddof=0) or 1.0)
        X = X.fillna(0.0)  # 0 == training centre after scaling.
    else:
        for a in X.columns:
            col = X[a]
            if col.min(skipna=True) >= 0 and col.skew(skipna=True) > log_skew_threshold:
                X[a] = np.log1p(col)
                logged.append(a)
        X = X.fillna(X.median())
        std = X.std(ddof=0)
        for a in std[std <= 1e-12].index:
            dropped[a] = "zero variance"
        X = X.loc[:, std > 1e-12]
        X = (X - X.mean()) / X.std(ddof=0)

    return {"X": X, "raw": raw, "meta": meta, "logged": logged, "dropped": dropped}


def fit_static_pca(X: pd.DataFrame, var_target: float = 0.9, random_state: int = 0):
    """PCA keeping the fewest components explaining ``var_target`` of variance.

    Returns ``(Z, pca, loadings, explained)`` where ``loadings`` is attrs x PCs.
    """
    from sklearn.decomposition import PCA  # pylint: disable=g-import-not-at-top

    full = PCA(random_state=random_state).fit(X.values)
    n = int(np.searchsorted(np.cumsum(full.explained_variance_ratio_), var_target) + 1)
    n = max(2, min(n, X.shape[1]))
    pca = PCA(n_components=n, random_state=random_state).fit(X.values)
    cols = [f"PC{i + 1}" for i in range(n)]
    Z = pd.DataFrame(pca.transform(X.values), index=X.index, columns=cols)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=cols)
    explained = pd.Series(full.explained_variance_ratio_, index=[f"PC{i + 1}" for i in range(len(full.explained_variance_ratio_))])
    return Z, pca, loadings, explained


def describe_pcs(loadings: pd.DataFrame, n_pcs: int = 5, top: int = 4) -> pd.DataFrame:
    """Top +/- loading attributes per PC, for naming components."""
    rows = []
    for pc in loadings.columns[:n_pcs]:
        s = loadings[pc].sort_values()
        rows.append({
            "PC": pc,
            "High (+)": ", ".join(s.index[::-1][:top]),
            "Low (-)": ", ".join(s.index[:top]),
        })
    return pd.DataFrame(rows).set_index("PC")


# =============================================================================
# 2. Targets
# =============================================================================
def parse_config_axes(config_ids: Iterable[str]) -> pd.DataFrame:
    """Parses hyperparameter axes (target, window, loss, bg, epochs) from config IDs."""
    rows = {}
    for cid in config_ids:
        s = str(cid)
        tgt = re.search(r"_(HS|H|S|D|T|A)(?:_[a-z]+)?_w\d", s)
        win = re.search(r"_w(\d+)", s)
        loss = re.search(r"_(NSE|MSE|KGE)(?:_|$)", s)
        bg = re.search(r"_bg([0-9.eE-]+?)(?:_|$)", s)
        ep = re.search(r"_ep(\d+)", s)
        lr = re.search(r"_lrs([0-9.eE-]+?)(?:_|$)", s) or re.search(r"_lr([0-9.eE-]+?)(?:_|$)", s)
        tol = re.search(r"_tol([0-9.]+|None)", s)
        rows[s] = {
            "Target": tgt.group(1) if tgt else "?",
            "Window": f"w{win.group(1)}" if win else "?",
            "Loss": loss.group(1) if loss else "?",
            "BG Weight": f"bg{bg.group(1)}" if bg else "?",
            "LR": f"lr{lr.group(1)}" if lr else "?",
            "Tol": f"tol{tol.group(1)}" if tol else "tolNone",
            "Epochs": f"ep{ep.group(1)}" if ep else "?",
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def short_config_labels(config_ids: Sequence[str]) -> Dict[str, str]:
    """Compact, unique display labels, e.g. ``'W2B_08 | HS w14 NSE bg0.5'``."""
    axes = parse_config_axes(config_ids)
    labels = {}
    for cid in config_ids:
        prefix = re.split(r"_(?:HS|H|S|D|T|A)(?:_[a-z]+)?_w\d", str(cid))[0]
        a = axes.loc[str(cid)]
        labels[cid] = f"{prefix} | {a['Target']} {a['Window']} {a['Loss']} {a['BG Weight']}"
    counts = pd.Series(list(labels.values())).value_counts()
    for cid in config_ids:
        if counts[labels[cid]] > 1:
            labels[cid] = f"{labels[cid]} {axes.loc[str(cid), 'Epochs']}"
    return labels


def build_config_regret_matrix(
    df_eval: pd.DataFrame,
    lead_times: Sequence[int] = (1,),
    metric_col: str = "NSE Skill Score",
    min_gap: float = 0.02,
    exclude_config_patterns: Sequence[str] = DEFAULT_EXCLUDE_CONFIG_PATTERNS,
    include_configs: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Builds the basin x config skill matrix and derived regret targets.

    Skill is averaged over ``lead_times``. Only basins with a finite value for
    every real configuration are kept (synthetic Per-Basin/Baseline rows dropped).
    """
    df = _ensure_skill_score_columns(df_eval.copy())
    df = df[df["Lead Time (Days)"].isin(list(lead_times))]
    excl = tuple(SYNTHETIC_CONFIG_PATTERNS) + tuple(p.lower() for p in exclude_config_patterns)
    synth = df["Config ID"].astype(str).str.lower().apply(lambda c: any(p in c for p in excl))
    df = df[~synth]
    if include_configs:
        df = df[df["Config ID"].isin(list(include_configs))]
    S = df.pivot_table(index="Basin ID", columns="Config ID", values=metric_col, aggfunc="mean")
    S = S.replace([np.inf, -np.inf], np.nan)
    n_all = len(S)
    S = S.dropna(axis=0, how="any")
    if S.empty:
        raise ValueError("No basin has finite skill for every configuration.")

    regret = S.max(axis=1).values[:, None] - S
    srt = np.sort(S.values, axis=1)
    gap = pd.Series(srt[:, -1] - srt[:, -2], index=S.index, name="winner_gap")
    winner = S.idxmax(axis=1).rename("winner")
    confident = winner.where(gap >= min_gap).rename("confident_winner")

    axes = parse_config_axes(S.columns)
    axis_winners, axis_gaps = {}, {}
    for ax in axes.columns:
        levels = axes[ax].unique()
        if len(levels) < 2:
            continue
        best_per_level = pd.DataFrame({lv: S[axes.index[axes[ax] == lv]].max(axis=1) for lv in levels})
        axis_winners[ax] = best_per_level.idxmax(axis=1)
        v = np.sort(best_per_level.values, axis=1)
        axis_gaps[ax] = pd.Series(v[:, -1] - v[:, -2], index=S.index)

    return {
        "S": S,
        "regret": regret,
        "winner": winner,
        "winner_gap": gap,
        "confident_winner": confident,
        "axis_winners": pd.DataFrame(axis_winners),
        "axis_gaps": pd.DataFrame(axis_gaps),
        "axes": axes,
        "n_basins_total": n_all,
        "lead_times": tuple(lead_times),
        "metric_col": metric_col,
        "min_gap": min_gap,
    }


# =============================================================================
# 3A. Unsupervised clustering + association tests
# =============================================================================
def cluster_static_space(
    Z: pd.DataFrame,
    k: Optional[int] = None,
    k_range: Iterable[int] = range(3, 13),
    method: str = "kmeans",
    random_state: int = 0,
):
    """Clusters the PCA space. Picks k by silhouette (kmeans) or BIC (gmm) when ``k`` is None.

    Returns ``(labels, diagnostics_df, chosen_k)``.
    """
    from sklearn.cluster import KMeans  # pylint: disable=g-import-not-at-top
    from sklearn.metrics import silhouette_score  # pylint: disable=g-import-not-at-top
    from sklearn.mixture import GaussianMixture  # pylint: disable=g-import-not-at-top

    def _fit(kk):
        if method == "gmm":
            m = GaussianMixture(n_components=kk, covariance_type="full", random_state=random_state, n_init=3).fit(Z.values)
            return m, m.predict(Z.values)
        m = KMeans(n_clusters=kk, n_init=10, random_state=random_state).fit(Z.values)
        return m, m.labels_

    diag = []
    ks = [k] if k else [kk for kk in k_range if 2 <= kk < len(Z)]
    fits = {}
    for kk in ks:
        m, lab = _fit(kk)
        fits[kk] = lab
        diag.append({
            "k": kk,
            "silhouette": float(silhouette_score(Z.values, lab)) if len(set(lab)) > 1 else np.nan,
            "bic": float(m.bic(Z.values)) if method == "gmm" else np.nan,
            "inertia": float(getattr(m, "inertia_", np.nan)),
        })
    diag = pd.DataFrame(diag).set_index("k")
    if k:
        chosen = k
    elif method == "gmm":
        chosen = int(diag["bic"].idxmin())
    else:
        chosen = int(diag["silhouette"].idxmax())
    return pd.Series(fits[chosen], index=Z.index, name="cluster"), diag, chosen


def _cramers_v(a: pd.Series, b: pd.Series) -> float:
    from scipy.stats import chi2_contingency  # pylint: disable=g-import-not-at-top

    ct = pd.crosstab(a, b)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return np.nan
    chi2 = chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def test_cluster_winner_association(
    labels: pd.Series, winner: pd.Series, n_perm: int = 1000, random_state: int = 0
) -> Dict[str, float]:
    """Cramér's V between cluster and winner with a label-permutation p-value."""
    idx = winner.dropna().index.intersection(labels.index)
    a, b = labels.loc[idx], winner.loc[idx]
    if len(idx) < 10:
        return {"n": len(idx), "cramers_v": np.nan, "null_mean": np.nan, "p_value": np.nan}
    obs = _cramers_v(a, b)
    rng = np.random.default_rng(random_state)
    null = np.array([_cramers_v(a, pd.Series(rng.permutation(b.values), index=idx)) for _ in range(n_perm)])
    return {
        "n": int(len(idx)),
        "cramers_v": obs,
        "null_mean": float(np.nanmean(null)),
        "p_value": float((np.sum(null >= obs) + 1) / (n_perm + 1)),
    }


def summarize_clusters(
    labels: pd.Series, regret: pd.DataFrame, X: pd.DataFrame, top_attrs: int = 3
) -> pd.DataFrame:
    """Per-cluster N, best config (min mean regret), and the attributes that most distinguish it."""
    labs = short_config_labels(list(regret.columns))
    global_best = regret.mean().idxmin()
    rows = []
    for c in sorted(labels.unique()):
        idx = labels.index[labels == c]
        mr = regret.loc[idx].mean()
        prof = X.loc[idx].mean().sort_values()
        rows.append({
            "Cluster": c,
            "N": len(idx),
            "Cluster-Best Config": labs[mr.idxmin()],
            "Mean Regret (cluster best)": mr.min(),
            "Mean Regret (global best)": mr[global_best],
            "High attrs (z)": ", ".join(f"{a} {prof[a]:+.1f}" for a in prof.index[::-1][:top_attrs]),
            "Low attrs (z)": ", ".join(f"{a} {prof[a]:+.1f}" for a in prof.index[:top_attrs]),
        })
    return pd.DataFrame(rows).set_index("Cluster")


# =============================================================================
# 3B/C. Policy value under spatially grouped CV
# =============================================================================
def make_cv_groups(meta: pd.DataFrame, group_col: str = "spatial_block", n_blocks: int = 20, random_state: int = 0) -> pd.Series:
    """Grouping for CV. ``spatial_block`` = KMeans on unit-sphere lat/lon (balanced spatial blocks)."""
    if group_col in ("country", "wmo_reg") and group_col in meta:
        return meta[group_col].astype(str)
    if group_col == "spatial_block":
        from sklearn.cluster import KMeans  # pylint: disable=g-import-not-at-top

        lat = np.radians(meta["lat"].fillna(0).values)
        lon = np.radians(meta["lon"].fillna(0).values)
        xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        nb = min(n_blocks, len(meta))
        return pd.Series(KMeans(n_clusters=nb, n_init=10, random_state=random_state).fit_predict(xyz), index=meta.index).astype(str)
    return pd.Series(np.arange(len(meta)).astype(str), index=meta.index)  # "none" -> random folds


def _fold_iter(groups: pd.Series, n_splits: int, random_state: int):
    from sklearn.model_selection import GroupKFold, KFold  # pylint: disable=g-import-not-at-top

    n_groups = groups.nunique()
    if n_groups >= n_splits and n_groups < len(groups):
        return GroupKFold(n_splits=n_splits).split(np.zeros(len(groups)), groups=groups.values)
    return KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.zeros(len(groups)))


def _choose_min_mean_regret(R: np.ndarray) -> int:
    return int(np.argmin(R.mean(axis=0)))


def _fit_regret_models(F_tr: np.ndarray, R_tr: np.ndarray, random_state: int):
    from sklearn.ensemble import HistGradientBoostingRegressor  # pylint: disable=g-import-not-at-top

    return [
        HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=random_state).fit(F_tr, R_tr[:, j])
        for j in range(R_tr.shape[1])
    ]


def _predict_choice(models, F: np.ndarray) -> np.ndarray:
    return np.argmin(np.column_stack([m.predict(F) for m in models]), axis=1)


def evaluate_assignment_policies(
    S: pd.DataFrame,
    Z: pd.DataFrame,
    groups: pd.Series,
    k: int,
    n_splits: int = 5,
    knn_k: int = 25,
    min_cluster_n: int = 20,
    include_shuffled_control: bool = True,
    random_state: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Out-of-fold evaluation of config-assignment policies.

    Policies (all choices made on training folds only):
      * Global Best - single config minimizing mean regret on the train fold.
      * Cluster Best - KMeans(k) on train PCs; each cluster uses its own best config
        (clusters with < ``min_cluster_n`` train basins fall back to Global).
      * kNN Best - best config among the ``knn_k`` nearest train basins in PC space.
      * Predicted Best - per-config HistGradientBoosting regret regressors; pick argmin.
      * Oracle - per-basin best (upper bound).
    Controls re-run Cluster/Predicted with attribute rows shuffled across basins.

    Returns ``(policy_df, choices_df)``.
    """
    from sklearn.cluster import KMeans  # pylint: disable=g-import-not-at-top
    from sklearn.neighbors import NearestNeighbors  # pylint: disable=g-import-not-at-top

    idx = S.index.intersection(Z.index).intersection(groups.index)
    S, Z, groups = S.loc[idx], Z.loc[idx], groups.loc[idx]
    Sv = S.values
    R = Sv.max(axis=1, keepdims=True) - Sv
    F = Z.values
    rng = np.random.default_rng(random_state)
    F_shuf = F[rng.permutation(len(F))]

    choice = {p: np.full(len(S), -1) for p in ["Global Best", "Cluster Best", "kNN Best", "Predicted Best"]}
    if include_shuffled_control:
        choice["Cluster Best (shuffled attrs)"] = np.full(len(S), -1)
        choice["Predicted Best (shuffled attrs)"] = np.full(len(S), -1)

    def _cluster_policy(Ftr, Fte, Rtr, g_best):
        km = KMeans(n_clusters=min(k, len(Ftr)), n_init=10, random_state=random_state).fit(Ftr)
        lab_te = km.predict(Fte)
        out = np.full(len(Fte), g_best)
        for c in np.unique(lab_te):
            m_tr = km.labels_ == c
            if m_tr.sum() >= min_cluster_n:
                out[lab_te == c] = _choose_min_mean_regret(Rtr[m_tr])
        return out

    for tr, te in _fold_iter(groups, n_splits, random_state):
        g_best = _choose_min_mean_regret(R[tr])
        choice["Global Best"][te] = g_best
        choice["Cluster Best"][te] = _cluster_policy(F[tr], F[te], R[tr], g_best)
        nn = NearestNeighbors(n_neighbors=min(knn_k, len(tr))).fit(F[tr])
        nbrs = nn.kneighbors(F[te], return_distance=False)
        choice["kNN Best"][te] = np.argmin(R[tr][nbrs].mean(axis=1), axis=1)
        choice["Predicted Best"][te] = _predict_choice(_fit_regret_models(F[tr], R[tr], random_state), F[te])
        if include_shuffled_control:
            choice["Cluster Best (shuffled attrs)"][te] = _cluster_policy(F_shuf[tr], F_shuf[te], R[tr], g_best)
            choice["Predicted Best (shuffled attrs)"][te] = _predict_choice(_fit_regret_models(F_shuf[tr], R[tr], random_state), F_shuf[te])

    rows_i = np.arange(len(S))
    attained = {p: Sv[rows_i, c] for p, c in choice.items()}
    attained["Oracle"] = Sv.max(axis=1)
    g_mean, o_mean = attained["Global Best"].mean(), attained["Oracle"].mean()
    g_med, o_med = np.median(attained["Global Best"]), np.median(attained["Oracle"])
    order = ["Global Best", "Cluster Best", "kNN Best", "Predicted Best", "Oracle",
             "Cluster Best (shuffled attrs)", "Predicted Best (shuffled attrs)"]
    rows = []
    for p in order:
        if p not in attained:
            continue
        v = attained[p]
        rows.append({
            "Policy": p,
            "Mean SS": v.mean(),
            "Median SS": np.median(v),
            "Mean Regret": (attained["Oracle"] - v).mean(),
            "% Gap Recovered (mean)": 100 * (v.mean() - g_mean) / (o_mean - g_mean) if o_mean > g_mean else np.nan,
            "% Gap Recovered (median)": 100 * (np.median(v) - g_med) / (o_med - g_med) if o_med > g_med else np.nan,
            "% Basins Better than Global": 100 * np.mean(v > attained["Global Best"] + 1e-12),
            "% Basins Worse than Global": 100 * np.mean(v < attained["Global Best"] - 1e-12),
            "Is Control": "shuffled" in p,
        })
    policy_df = pd.DataFrame(rows).set_index("Policy")
    choices_df = pd.DataFrame({p: S.columns[c] for p, c in choice.items()}, index=S.index)
    return policy_df, choices_df


def compute_attribute_importance(
    X: pd.DataFrame,
    S: pd.DataFrame,
    groups: pd.Series,
    n_repeats: int = 5,
    random_state: int = 0,
) -> pd.DataFrame:
    """Grouped hold-out permutation importance on *raw standardized attributes*.

    Importance = increase in held-out mean regret of the Predicted-Best policy when
    one attribute column is shuffled (SS units; >0 means the attribute helps choose).
    """
    from sklearn.model_selection import GroupShuffleSplit  # pylint: disable=g-import-not-at-top

    idx = S.index.intersection(X.index).intersection(groups.index)
    Xv, Sv = X.loc[idx].values, S.loc[idx].values
    R = Sv.max(axis=1, keepdims=True) - Sv
    g = groups.loc[idx].values
    if len(np.unique(g)) >= 4:
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=random_state).split(Xv, groups=g))
    else:
        perm = np.random.default_rng(random_state).permutation(len(idx))
        cut = int(0.7 * len(idx))
        tr, te = perm[:cut], perm[cut:]
    models = _fit_regret_models(Xv[tr], R[tr], random_state)
    rows_te = np.arange(len(te))
    base = R[te][rows_te, _predict_choice(models, Xv[te])].mean()
    rng = np.random.default_rng(random_state)
    out = []
    for j, a in enumerate(X.columns):
        incs = []
        for _ in range(n_repeats):
            Xp = Xv[te].copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            incs.append(R[te][rows_te, _predict_choice(models, Xp)].mean() - base)
        out.append({"Attribute": a, "Importance": float(np.mean(incs)), "Std": float(np.std(incs))})
    return pd.DataFrame(out).set_index("Attribute").sort_values("Importance", ascending=False)


# =============================================================================
# Orchestrator
# =============================================================================
def run_config_affinity(
    df_eval: pd.DataFrame,
    lead_times: Sequence[int] = (1,),
    metric_col: str = "NSE Skill Score",
    k: Optional[int] = None,
    method: str = "kmeans",
    min_gap: float = 0.02,
    var_target: float = 0.9,
    group_col: str = "spatial_block",
    n_splits: int = 5,
    n_perm: int = 1000,
    use_model_scaler: bool = False,
    attr_zarr_path: Optional[str | Path] = None,
    compute_importance: bool = True,
    random_state: int = 0,
    verbose: bool = True,
) -> Dict[str, object]:
    """Runs the full Stage 6b analysis and returns every intermediate artefact."""
    tgt = build_config_regret_matrix(df_eval, lead_times=lead_times, metric_col=metric_col, min_gap=min_gap)
    feats = build_static_feature_matrix(list(tgt["S"].index), attr_zarr_path=attr_zarr_path, use_model_scaler=use_model_scaler)
    X = feats["X"]
    common = tgt["S"].index.intersection(X.index)
    S, regret = tgt["S"].loc[common], tgt["regret"].loc[common]
    X, meta = X.loc[common], feats["meta"].loc[common]
    Z, pca, loadings, explained = fit_static_pca(X, var_target=var_target, random_state=random_state)
    labels, k_diag, k_chosen = cluster_static_space(Z, k=k, method=method, random_state=random_state)
    groups = make_cv_groups(meta, group_col=group_col, random_state=random_state)

    assoc = {
        "cluster x winner": test_cluster_winner_association(labels, tgt["winner"].loc[common], n_perm, random_state),
        "cluster x confident winner": test_cluster_winner_association(labels, tgt["confident_winner"].loc[common], n_perm, random_state),
    }
    for ax in tgt["axis_winners"].columns:
        conf_ax = tgt["axis_winners"][ax].where(tgt["axis_gaps"][ax] >= min_gap).loc[common]
        assoc[f"cluster x {ax} (confident)"] = test_cluster_winner_association(labels, conf_ax, n_perm, random_state)
    assoc_df = pd.DataFrame(assoc).T

    policy_df, choices_df = evaluate_assignment_policies(S, Z, groups, k=k_chosen, n_splits=n_splits, random_state=random_state)
    importance = compute_attribute_importance(X, S, groups, random_state=random_state) if compute_importance else pd.DataFrame()

    res = {
        **tgt,
        "S": S, "regret": regret, "X": X, "raw": feats["raw"].loc[common], "meta": meta,
        "logged": feats["logged"], "dropped": feats["dropped"],
        "Z": Z, "pca": pca, "loadings": loadings, "explained": explained, "pc_names": describe_pcs(loadings),
        "labels": labels, "k": k_chosen, "k_diagnostics": k_diag, "method": method,
        "cluster_summary": summarize_clusters(labels, regret, X),
        "groups": groups, "group_col": group_col,
        "association_df": assoc_df, "policy_df": policy_df, "choices_df": choices_df,
        "importance": importance, "config_labels": short_config_labels(list(S.columns)),
    }
    if verbose:
        pr = policy_df
        print(
            f"[Stage 6b] Leads {list(lead_times)} | {len(common):,}/{tgt['n_basins_total']:,} basins complete | "
            f"{X.shape[1]} attrs -> {Z.shape[1]} PCs ({explained.iloc[:Z.shape[1]].sum():.0%} var) | k={k_chosen} ({method}) | "
            f"CV groups: {group_col} ({groups.nunique()})"
        )
        print(
            f"           Gap recovered (mean SS): Cluster {pr.loc['Cluster Best', '% Gap Recovered (mean)']:.1f}% | "
            f"kNN {pr.loc['kNN Best', '% Gap Recovered (mean)']:.1f}% | Predicted {pr.loc['Predicted Best', '% Gap Recovered (mean)']:.1f}% | "
            f"shuffled control {pr.loc['Predicted Best (shuffled attrs)', '% Gap Recovered (mean)']:.1f}%"
        )
    return res


def interpret_affinity(res: Dict[str, object]) -> str:
    """One-paragraph verdict following the plan's decision table."""
    pr = res["policy_df"]
    best_real = pr.loc[["Cluster Best", "kNN Best", "Predicted Best"], "% Gap Recovered (mean)"].max()
    ctrl = pr.loc[[p for p in pr.index if "shuffled" in p], "% Gap Recovered (mean)"].max()
    p_conf = res["association_df"].loc["cluster x confident winner", "p_value"]
    ax_rows = res["association_df"].loc[[r for r in res["association_df"].index if r.startswith("cluster x ") and "(confident)" in r]]
    strong_ax = ax_rows[(ax_rows["p_value"] < 0.01)].sort_values("cramers_v", ascending=False)
    ax_txt = (", ".join(f"{r.replace('cluster x ', '').replace(' (confident)', '')} (V={v:.2f})" for r, v in strong_ax["cramers_v"].items())
              or "none")
    if best_real >= 30 and p_conf < 0.01:
        verdict = "REAL STRUCTURE: static attributes meaningfully predict the best configuration."
    elif best_real >= 10:
        verdict = "PARTIAL STRUCTURE: attributes recover some of the gap; consider choosing only the associated axis per basin."
    else:
        verdict = "NO USABLE STRUCTURE: per-basin winners are mostly noise w.r.t. static attributes; keep a single Global Best."
    return (
        f"{verdict}\n  Best attribute-driven policy recovers {best_real:.1f}% of the Global->Oracle mean-SS gap "
        f"(shuffled-attribute control: {ctrl:.1f}%). Cluster x confident-winner p = {p_conf:.3g}. "
        f"Hyperparameter axes significantly associated with clusters (p<0.01): {ax_txt}."
    )


# =============================================================================
# 4. Plots
# =============================================================================
def _is_nse_ss(res: Dict[str, object]) -> bool:
    """True when the affinity analysis ran on the NSE skill score (SS display units apply)."""
    col = str(res.get("metric_col", "NSE Skill Score"))
    return "skill" in col.lower() and "kge" not in col.lower()


def _gap_txt(res: Dict[str, object]) -> str:
    """``min_gap`` as display text: '2%' in SS-percent mode, otherwise unchanged (e.g. '0.02')."""
    g = res["min_gap"]
    return f"{float(g) * 100:g}%" if (units.ss_percent() and _is_nse_ss(res)) else f"{g}"


def _ss_txt(res: Dict[str, object], v: float, frac_fmt: str = ".3f") -> str:
    """Unsigned SS-unit text for annotations; exact legacy ``frac_fmt`` when not in percent mode."""
    return units.fmt_ss(v, sign=False) if (units.ss_percent() and _is_nse_ss(res)) else f"{v:{frac_fmt}}"


def _config_palette(configs: Sequence[str]) -> Dict[str, str]:
    return {c: QUAL_COLORS[i % len(QUAL_COLORS)] for i, c in enumerate(sorted(configs))}


def plot_pca_biplot(res: Dict[str, object], color_by: str = "confident_winner", n_arrows: int = 8, figsize=(15, 6)):
    """Scree (left) + PC1/PC2 biplot coloured by (confident) winner with top loading arrows (right)."""
    Z, loadings, explained = res["Z"], res["loadings"], res["explained"]
    labs = res["config_labels"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [1, 2.2]})
    n_show = min(max(20, Z.shape[1] + 5), len(explained))
    ax0.bar(range(1, n_show + 1), explained.values[:n_show], color="#9ecae1", label="Per PC")
    ax0.plot(range(1, n_show + 1), np.cumsum(explained.values[:n_show]), "o-", color="#08519c", ms=3, label="Cumulative")
    ax0.axvline(Z.shape[1], color="k", ls="--", lw=1)
    ax0.set(xlabel="Principal component", ylabel="Explained variance", title=f"(a) Scree: {Z.shape[1]} PCs retained")
    ax0.legend(frameon=False, fontsize=8)

    if color_by == "cluster":
        lab = res["labels"]
        for i, c in enumerate(sorted(lab.unique())):
            m = lab == c
            ax1.scatter(Z.loc[m, "PC1"], Z.loc[m, "PC2"], s=10, alpha=0.7, color=QUAL_COLORS[i % 12], label=f"Cluster {c} (n={m.sum()})")
    else:
        w = res[color_by].reindex(Z.index)
        pal = _config_palette(res["S"].columns)
        m_na = w.isna()
        ax1.scatter(Z.loc[m_na, "PC1"], Z.loc[m_na, "PC2"], s=6, color="#d9d9d9", alpha=0.5, label=f"Near-tie (gap<{_gap_txt(res)})")
        for c in w.dropna().value_counts().index:
            m = w == c
            ax1.scatter(Z.loc[m, "PC1"], Z.loc[m, "PC2"], s=14, alpha=0.85, color=pal[c], label=f"{labs[c]} (n={m.sum()})")
    top = loadings[["PC1", "PC2"]].pow(2).sum(axis=1).sort_values(ascending=False).index[:n_arrows]
    scale = 0.9 * np.abs(Z[["PC1", "PC2"]].values).max() / max(np.abs(loadings.loc[top, ["PC1", "PC2"]].values).max(), 1e-9)
    for a in top:
        x, y = loadings.loc[a, "PC1"] * scale, loadings.loc[a, "PC2"] * scale
        ax1.arrow(0, 0, x, y, color="k", alpha=0.6, width=0.02, head_width=0.25, length_includes_head=True)
        ax1.text(x * 1.07, y * 1.07, a, fontsize=8, ha="center", va="center", bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
    ax1.set(xlabel=f"PC1 ({explained['PC1']:.0%})", ylabel=f"PC2 ({explained['PC2']:.0%})",
            title=f"(b) Static-attribute PCA coloured by {color_by.replace('_', ' ')}")
    ax1.axhline(0, color="#bbb", lw=0.5); ax1.axvline(0, color="#bbb", lw=0.5)
    ax1.legend(fontsize=7, frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    return fig


def plot_cluster_regret_heatmap(res: Dict[str, object], figsize=None):
    """Cluster x config mean regret (SS units); each cluster's best config outlined, Global row on top."""
    regret, lab, labs = res["regret"], res["labels"], res["config_labels"]
    cols = list(regret.mean().sort_values().index)
    rows = {"ALL (global)": regret[cols].mean()}
    for c in sorted(lab.unique()):
        rows[f"C{c} (n={int((lab == c).sum())})"] = regret.loc[lab == c, cols].mean()
    H = pd.DataFrame(rows).T
    fig, ax = plt.subplots(figsize=figsize or (1.1 * len(cols) + 3, 0.45 * len(H) + 2.5))
    im = ax.imshow(H.values, cmap="YlOrRd", aspect="auto")
    for i in range(H.shape[0]):
        j = int(np.argmin(H.values[i]))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="#08519c", lw=2.5))
        for jj in range(H.shape[1]):
            ax.text(jj, i, _ss_txt(res, H.values[i, jj]), ha="center", va="center", fontsize=7,
                    color="white" if H.values[i, jj] > np.nanpercentile(H.values, 75) else "black")
    ax.axhline(0.5, color="k", lw=1.5)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([labs[c] for c in cols], rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(H))); ax.set_yticklabels(H.index, fontsize=8)
    is_ss = _is_nse_ss(res)
    cb = fig.colorbar(im, ax=ax, label=units.ss_label("Mean regret (SS_best − SS_config)") if is_ss else "Mean regret (SS_best − SS_config)", shrink=0.8)
    if is_ss:
        units.ss_colorbar(cb)
    ax.set_title("Cluster × Configuration mean regret (blue box = cluster-best; lower is better)")
    fig.tight_layout()
    return fig


def _load_world():
    try:
        import geopandas as gpd  # pylint: disable=g-import-not-at-top

        for p in COUNTRY_SHP_PATHS:
            if Path(p).exists():
                return gpd.read_file(p)
    except Exception:
        pass
    return None


def plot_cluster_map(res: Dict[str, object], color_by: str = "cluster", marker_size: float = 10.0, figsize=(15, 7)):
    """World map of basins coloured by static cluster (or by confident winner / chosen policy config)."""
    meta = res["meta"]
    fig, ax = plt.subplots(figsize=figsize)
    world = _load_world()
    if world is not None:
        world.plot(ax=ax, color="#f2f2f2", edgecolor="#bdbdbd", lw=0.4)
    if color_by == "cluster":
        lab = res["labels"]
        for i, c in enumerate(sorted(lab.unique())):
            m = lab.index[lab == c]
            ax.scatter(meta.loc[m, "lon"], meta.loc[m, "lat"], s=marker_size, color=QUAL_COLORS[i % 12], alpha=0.85,
                       label=f"C{c} (n={len(m)})", edgecolors="none")
    else:
        w = (res["choices_df"][color_by] if color_by in res["choices_df"] else res[color_by]).reindex(meta.index)
        pal, labs = _config_palette(res["S"].columns), res["config_labels"]
        m_na = w.isna()
        ax.scatter(meta.loc[m_na, "lon"], meta.loc[m_na, "lat"], s=marker_size * 0.6, color="#cccccc", alpha=0.6, label="Near-tie", edgecolors="none")
        for c in w.dropna().value_counts().index:
            m = w.index[w == c]
            ax.scatter(meta.loc[m, "lon"], meta.loc[m, "lat"], s=marker_size, color=pal[c], alpha=0.85, label=f"{labs[c]} (n={len(m)})", edgecolors="none")
    lon, lat = meta["lon"].dropna(), meta["lat"].dropna()
    ax.set_xlim(lon.min() - 5, lon.max() + 5); ax.set_ylim(lat.min() - 5, lat.max() + 5)
    ax.set(xlabel="Longitude", ylabel="Latitude", title=f"Basins coloured by {color_by.replace('_', ' ')}")
    ax.legend(fontsize=7, frameon=True, loc="lower left", markerscale=1.5)
    fig.tight_layout()
    return fig


def plot_policy_ladder(res: Dict[str, object], figsize=(12, 5)):
    """Mean SS and % of Global->Oracle gap recovered for each assignment policy (controls hatched)."""
    pr = res["policy_df"]
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
    colors = ["#bdbdbd" if c else "#4e79a7" for c in pr["Is Control"]]
    colors = ["#2ca02c" if p == "Oracle" else ("#7f7f7f" if p == "Global Best" else c) for p, c in zip(pr.index, colors)]
    y = np.arange(len(pr))
    ax0.barh(y, pr["Mean SS"], color=colors, hatch=None)
    for i, v in enumerate(pr["Mean SS"]):
        ax0.text(v, i, f" {_ss_txt(res, v)}", va="center", fontsize=8)
    lo = pr["Mean SS"].min()
    ax0.set_xlim(lo - 0.3 * (pr["Mean SS"].max() - lo) - 1e-3, pr["Mean SS"].max() + 0.25 * (pr["Mean SS"].max() - lo) + 1e-3)
    ax0.set_yticks(y); ax0.set_yticklabels(pr.index, fontsize=9); ax0.invert_yaxis()
    ax0.set(xlabel=f"Out-of-fold mean {res['metric_col']}", title="(a) Attained skill by policy")
    if _is_nse_ss(res):
        units.ss_axis(ax0, "x")
        ax0.set_xlabel(units.ss_label(ax0.get_xlabel()))
    gr = pr["% Gap Recovered (mean)"]
    bars = ax1.barh(y, gr, color=colors)
    for b, p in zip(bars, pr.index):
        if "shuffled" in p:
            b.set_hatch("//")
    for i, v in enumerate(gr):
        ax1.text(v, i, f" {v:.1f}%", va="center", fontsize=8)
    for thr, txt in [(10, "10%"), (30, "30%")]:
        ax1.axvline(thr, color="#d62728", ls=":", lw=1)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_yticks(y); ax1.set_yticklabels([]); ax1.invert_yaxis()
    ax1.set(xlabel="% of Global→Oracle gap recovered (mean SS)", title=f"(b) Policy value (CV groups: {res['group_col']})")
    fig.tight_layout()
    return fig


def plot_axis_affinity(res: Dict[str, object], figsize=None):
    """For each hyperparameter axis: PC1/PC2 scatter coloured by the confident best level of that axis."""
    axw, axg, Z = res["axis_winners"], res["axis_gaps"], res["Z"]
    assoc = res["association_df"]
    axes_list = list(axw.columns)
    n = len(axes_list)
    if n == 0:
        print("[INFO] No hyperparameter axis has >1 level.")
        return None
    fig, axs = plt.subplots(1, n, figsize=figsize or (4.6 * n, 4.4), squeeze=False)
    for ax, name in zip(axs[0], axes_list):
        w = axw[name].where(axg[name] >= res["min_gap"]).reindex(Z.index)
        m_na = w.isna()
        ax.scatter(Z.loc[m_na, "PC1"], Z.loc[m_na, "PC2"], s=5, color="#e0e0e0", alpha=0.5, label="Near-tie")
        for i, lv in enumerate(sorted(w.dropna().unique())):
            m = w == lv
            ax.scatter(Z.loc[m, "PC1"], Z.loc[m, "PC2"], s=10, alpha=0.8, color=QUAL_COLORS[i % 12], label=f"{lv} (n={m.sum()})")
        row = f"cluster x {name} (confident)"
        stat = f"V={assoc.loc[row, 'cramers_v']:.2f}, p={assoc.loc[row, 'p_value']:.3f}" if row in assoc.index else ""
        ax.set(title=f"{name}\n{stat}", xlabel="PC1", ylabel="PC2")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle(f"Best level per hyperparameter axis (confident: gap ≥ {_gap_txt(res)})", y=1.02)
    fig.tight_layout()
    return fig


def plot_attribute_importance(res: Dict[str, object], top: int = 15, figsize=(8, 6)):
    """Grouped hold-out permutation importance of raw attributes for choosing the config."""
    imp = res["importance"]
    if imp is None or len(imp) == 0:
        print("[INFO] Importance not computed.")
        return None
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(d.index, d["Importance"], xerr=d["Std"], color=["#4e79a7" if v > 0 else "#bdbdbd" for v in d["Importance"]])
    ax.axvline(0, color="k", lw=0.8)
    ax.set(xlabel="Δ held-out mean regret when shuffled (SS units)", title=f"Top {top} attributes for choosing the DA config")
    if _is_nse_ss(res) and units.ss_percent():
        units.ss_axis(ax, "x")
        ax.set_xlabel("Δ held-out mean regret when shuffled (SS, %)")
    fig.tight_layout()
    return fig


def style_policy_table(policy_df: pd.DataFrame):
    """Styled policy ladder table."""
    # SS columns: 2 dp in percent mode (same precision as the legacy 4-dp fraction), legacy format otherwise.
    ss_fmt = units.ss_formatter(decimals=2, sign=False) if units.ss_percent() else "{:.4f}"
    fmt = {
        "Mean SS": ss_fmt, "Median SS": ss_fmt, "Mean Regret": ss_fmt,
        "% Gap Recovered (mean)": "{:.1f}%", "% Gap Recovered (median)": "{:.1f}%",
        "% Basins Better than Global": "{:.1f}%", "% Basins Worse than Global": "{:.1f}%",
    }
    d = policy_df.drop(columns=["Is Control"], errors="ignore")
    return (d.style.format(fmt)
            .background_gradient(cmap="RdYlGn", subset=["% Gap Recovered (mean)"], vmin=-30, vmax=60)
            .set_caption("Out-of-fold config-assignment policies (choices made on training folds only)"))
