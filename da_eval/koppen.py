"""Köppen-Geiger climate classification (Beck et al., 2023, v2) for DA evaluation maps.

Source: Beck, H. E., McVicar, T. R., Vergopolan, N., et al. (2023). High-resolution (1 km)
Köppen-Geiger maps for 1901-2099 based on constrained CMIP6 projections. Scientific Data 10, 724.
https://doi.org/10.1038/s41597-023-02549-6  (data: https://doi.org/10.6084/m9.figshare.21789074)

Provides:
  * ``KG_CLASSES`` / ``KG_GROUPS``: code -> symbol / description / Beck RGB colours.
  * ``load_kg_raster``: the gridded map (uint8 codes 0..30, 0 = ocean / no data) via PIL.
  * ``kg_rgba_image``: an RGBA image for ``imshow`` underlays ("groups" or "classes" mode).
  * ``compute_basin_koppen``: per-basin majority class from Caravan catchment polygons (cached CSV).
  * ``draw_koppen_underlay``: one-call matplotlib underlay + legend for the dashboard maps.
"""
from __future__ import annotations

import os
import zipfile
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
KG_DATA_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "koppen_geiger"))
KG_ZIP_URL = "https://ndownloader.figshare.com/files/61012822"
CARAVAN_SHP_ROOT = "/usr/local/google/home/kruparell/Caravans_V2/shapefiles"
KG_CITATION = "Beck et al. (2023), Sci. Data 10, 724 — Köppen-Geiger 1991–2020"

# code: (symbol, description, (r, g, b))
KG_CLASSES: Dict[int, Tuple[str, str, Tuple[int, int, int]]] = {
    1: ("Af", "Tropical, rainforest", (0, 0, 255)),
    2: ("Am", "Tropical, monsoon", (0, 120, 255)),
    3: ("Aw", "Tropical, savannah", (70, 170, 250)),
    4: ("BWh", "Arid, desert, hot", (255, 0, 0)),
    5: ("BWk", "Arid, desert, cold", (255, 150, 150)),
    6: ("BSh", "Arid, steppe, hot", (245, 165, 0)),
    7: ("BSk", "Arid, steppe, cold", (255, 220, 100)),
    8: ("Csa", "Temperate, dry summer, hot summer", (255, 255, 0)),
    9: ("Csb", "Temperate, dry summer, warm summer", (200, 200, 0)),
    10: ("Csc", "Temperate, dry summer, cold summer", (150, 150, 0)),
    11: ("Cwa", "Temperate, dry winter, hot summer", (150, 255, 150)),
    12: ("Cwb", "Temperate, dry winter, warm summer", (100, 200, 100)),
    13: ("Cwc", "Temperate, dry winter, cold summer", (50, 150, 50)),
    14: ("Cfa", "Temperate, no dry season, hot summer", (200, 255, 80)),
    15: ("Cfb", "Temperate, no dry season, warm summer", (100, 255, 80)),
    16: ("Cfc", "Temperate, no dry season, cold summer", (50, 200, 0)),
    17: ("Dsa", "Cold, dry summer, hot summer", (255, 0, 255)),
    18: ("Dsb", "Cold, dry summer, warm summer", (200, 0, 200)),
    19: ("Dsc", "Cold, dry summer, cold summer", (150, 50, 150)),
    20: ("Dsd", "Cold, dry summer, very cold winter", (150, 100, 150)),
    21: ("Dwa", "Cold, dry winter, hot summer", (170, 175, 255)),
    22: ("Dwb", "Cold, dry winter, warm summer", (90, 120, 220)),
    23: ("Dwc", "Cold, dry winter, cold summer", (75, 80, 180)),
    24: ("Dwd", "Cold, dry winter, very cold winter", (50, 0, 135)),
    25: ("Dfa", "Cold, no dry season, hot summer", (0, 255, 255)),
    26: ("Dfb", "Cold, no dry season, warm summer", (55, 200, 255)),
    27: ("Dfc", "Cold, no dry season, cold summer", (0, 125, 125)),
    28: ("Dfd", "Cold, no dry season, very cold winter", (0, 70, 95)),
    29: ("ET", "Polar, tundra", (178, 178, 178)),
    30: ("EF", "Polar, frost", (102, 102, 102)),
}

# group letter: (name, pastel display colour)
KG_GROUPS: Dict[str, Tuple[str, str]] = {
    "A": ("Tropical", "#6fa8dc"),
    "B": ("Arid", "#f6b26b"),
    "C": ("Temperate", "#93c47d"),
    "D": ("Cold", "#b4a7d6"),
    "E": ("Polar", "#7f8fa6"),
}

KOPPEN_MODES = ("off", "groups", "classes")


def kg_group_of(code) -> Optional[str]:
    try:
        c = int(code)
    except (TypeError, ValueError):
        return None
    return KG_CLASSES[c][0][0] if c in KG_CLASSES else None


def ensure_kg_data(data_dir: str = KG_DATA_DIR, period: str = "1991_2020", res: str = "0p1") -> str:
    """Returns the path to the requested GeoTIFF, downloading/extracting the Beck archive if needed."""
    tif = os.path.join(data_dir, period, f"koppen_geiger_{res}.tif")
    if os.path.exists(tif):
        return tif
    os.makedirs(data_dir, exist_ok=True)
    zpath = os.path.join(data_dir, "koppen_geiger_tif.zip")
    if not os.path.exists(zpath):
        import urllib.request
        print(f"[koppen] Downloading Beck et al. (2023) Köppen-Geiger maps -> {zpath}")
        urllib.request.urlretrieve(KG_ZIP_URL, zpath)
    with zipfile.ZipFile(zpath) as zf:
        members = [m for m in zf.namelist() if m.startswith(f"{period}/") or m.endswith("legend.txt")]
        zf.extractall(data_dir, members=members)
    if not os.path.exists(tif):
        raise FileNotFoundError(tif)
    return tif


_RASTER_CACHE: Dict[Tuple[str, str], Tuple[np.ndarray, List[float]]] = {}


def load_kg_raster(res: str = "0p1", period: str = "1991_2020") -> Tuple[np.ndarray, List[float]]:
    """Loads the global KG grid as uint8 codes (row 0 = 90°N). Returns (array, [x0, x1, y0, y1])."""
    key = (res, period)
    if key not in _RASTER_CACHE:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None  # the 1-km grid is 43200 x 21600
        with Image.open(ensure_kg_data(period=period, res=res)) as im:
            arr = np.asarray(im, dtype=np.uint8)
        _RASTER_CACHE[key] = (arr, [-180.0, 180.0, -90.0, 90.0])
    return _RASTER_CACHE[key]


def kg_rgba_image(mode: str = "groups", alpha: float = 0.4, res: str = "0p1",
                  groups: Optional[Iterable[str]] = None) -> Tuple[np.ndarray, List[float]]:
    """RGBA underlay image. ``groups`` optionally restricts colouring to those KG groups."""
    arr, extent = load_kg_raster(res=res)
    lut = np.zeros((256, 4), dtype=np.float32)
    keep = {g.upper() for g in groups} if groups else None
    for code, (sym, _desc, rgb) in KG_CLASSES.items():
        g = sym[0]
        if keep is not None and g not in keep:
            continue
        if mode == "classes":
            lut[code, :3] = np.asarray(rgb) / 255.0
        else:
            h = KG_GROUPS[g][1].lstrip("#")
            lut[code, :3] = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
        lut[code, 3] = alpha
    return lut[arr], extent


def kg_legend_handles(mode: str = "groups", groups: Optional[Iterable[str]] = None,
                      present_codes: Optional[Iterable[int]] = None, alpha: float = 1.0) -> list:
    """Legend patches for the KG underlay (groups, or only the classes in ``present_codes``)."""
    from matplotlib.patches import Patch
    keep = {g.upper() for g in groups} if groups else None
    a = min(1.0, max(0.35, float(alpha)))  # match the map tint but keep swatches readable
    if mode == "classes":
        codes = sorted({int(c) for c in present_codes if pd.notna(c) and int(c) in KG_CLASSES}) \
            if present_codes is not None else sorted(KG_CLASSES)
        if keep is not None:
            codes = [c for c in codes if KG_CLASSES[c][0][0] in keep]
        return [Patch(facecolor=np.asarray(KG_CLASSES[c][2]) / 255.0, alpha=a, edgecolor="#5f6368", linewidth=0.4,
                      label=f"{KG_CLASSES[c][0]}") for c in codes]
    return [Patch(facecolor=col, alpha=a, edgecolor="#5f6368", linewidth=0.4, label=f"{g} {name}")
            for g, (name, col) in KG_GROUPS.items() if keep is None or g in keep]


def draw_koppen_underlay(ax, mode: str = "groups", alpha: float = 0.4, groups: Optional[Iterable[str]] = None,
                         present_codes: Optional[Iterable[int]] = None, legend: bool = True,
                         legend_loc: str = "lower left", world_gdf=None, font_scale: float = 1.0) -> None:
    """Draws the KG map under the catchment points (zorder 1.5) plus a compact legend."""
    mode = str(mode or "off").lower()
    if mode not in ("groups", "classes"):
        return
    try:
        img, extent = kg_rgba_image(mode=mode, alpha=alpha, groups=groups)
    except Exception as e:  # pragma: no cover - data missing / offline
        print(f"[koppen] Underlay unavailable: {e}")
        return
    ax.imshow(img, extent=extent, origin="upper", interpolation="nearest", zorder=1.5, aspect="auto")
    if world_gdf is not None:  # re-draw borders on top of the tint
        world_gdf.boundary.plot(ax=ax, color="#9aa0a6", linewidth=0.4, zorder=1.6)
    if not legend:
        return
    handles = kg_legend_handles(mode, groups, present_codes, alpha)
    if not handles:
        return
    ncol = max(1, int(np.ceil(len(handles) / 8))) if mode == "classes" else 1
    leg = ax.legend(handles=handles, loc=legend_loc, fontsize=7.5 * font_scale, ncol=ncol, frameon=True,
                    framealpha=0.9, title="Köppen-Geiger", title_fontsize=8 * font_scale, handlelength=1.2,
                    columnspacing=0.8)
    ax.add_artist(leg)  # keep it when later ax.legend() calls (selected-basin star) replace legend_


def _majority(codes: np.ndarray) -> Tuple[int, float]:
    codes = codes[codes > 0]
    if codes.size == 0:
        return 0, np.nan
    counts = np.bincount(codes, minlength=31)
    c = int(counts.argmax())
    return c, float(counts[c] / codes.size)


def compute_basin_koppen(basin_ids: Iterable[str], lat_lon: Optional[pd.DataFrame] = None,
                         cache_csv: Optional[str] = None, res: str = "0p00833333",
                         shp_root: str = CARAVAN_SHP_ROOT, verbose: bool = True) -> pd.DataFrame:
    """Majority KG class per basin over its Caravan catchment polygon (1-km grid by default).

    Basins whose polygon contains no cell centre (tiny catchments) or that have no polygon fall back to
    the class at the gauge ``lat``/``lon`` (from ``lat_lon``), then to the nearest land cell.
    Results are cached in ``cache_csv`` and only missing basins are computed on subsequent calls.
    Returns columns: Basin ID, kg_code, kg_symbol, kg_group, kg_group_name, kg_frac, kg_method.
    """
    cache_csv = cache_csv or os.path.join(KG_DATA_DIR, f"basin_koppen_1991_2020_{res}.csv")
    wanted = sorted({str(b) for b in basin_ids})
    cached = pd.read_csv(cache_csv, dtype={"Basin ID": str}) if os.path.exists(cache_csv) else pd.DataFrame()
    have = set(cached["Basin ID"]) if not cached.empty else set()
    todo = [b for b in wanted if b not in have]
    if todo:
        import geopandas as gpd
        import shapely
        arr, _ = load_kg_raster(res=res)
        ny, nx = arr.shape
        dx, dy = 360.0 / nx, 180.0 / ny

        def _cell_at(lon, lat) -> int:
            i = int(np.clip((90.0 - lat) / dy, 0, ny - 1))
            j = int(np.clip((lon + 180.0) / dx, 0, nx - 1))
            c = int(arr[i, j])
            r = 0
            while c == 0 and r < 5:  # coastal gauge in an ocean cell: widen the window
                r += 1
                win = arr[max(0, i - r):i + r + 1, max(0, j - r):j + r + 1]
                c, _ = _majority(win.ravel())
            return c

        ll = {}
        if lat_lon is not None and not lat_lon.empty:
            d = lat_lon.dropna(subset=["lat", "lon"]).drop_duplicates("Basin ID")
            ll = dict(zip(d["Basin ID"].astype(str), zip(d["lon"].astype(float), d["lat"].astype(float))))

        by_ds: Dict[str, List[str]] = {}
        for b in todo:
            by_ds.setdefault(b.split("_", 1)[0], []).append(b)
        rows = []
        for ds, basins in by_ds.items():
            shp = os.path.join(shp_root, ds, f"{ds}_basin_shapes.shp")
            geoms = {}
            if os.path.exists(shp):
                g = gpd.read_file(shp)
                g = g[g["gauge_id"].astype(str).isin(set(basins))]
                geoms = dict(zip(g["gauge_id"].astype(str), g.geometry))
            if verbose:
                print(f"[koppen] {ds}: {len(basins)} basins ({len(geoms)} polygons)")
            for b in basins:
                code, frac, method = 0, np.nan, "none"
                geom = geoms.get(b)
                if geom is not None and not geom.is_empty:
                    x0, y0, x1, y1 = geom.bounds
                    j0, j1 = int(np.floor((x0 + 180) / dx)), int(np.ceil((x1 + 180) / dx))
                    i0, i1 = int(np.floor((90 - y1) / dy)), int(np.ceil((90 - y0) / dy))
                    j0, i0 = max(j0, 0), max(i0, 0)
                    j1, i1 = min(j1, nx), min(i1, ny)
                    sub = arr[i0:i1, j0:j1]
                    if sub.size:
                        xs = -180 + (np.arange(j0, j1) + 0.5) * dx
                        ys = 90 - (np.arange(i0, i1) + 0.5) * dy
                        X, Y = np.meshgrid(xs, ys)
                        inside = shapely.contains_xy(geom, X, Y)
                        if inside.any():
                            code, frac = _majority(sub[inside])
                            method = "polygon"
                    if code == 0:
                        c = geom.representative_point()
                        code, method = _cell_at(c.x, c.y), "polygon_point"
                if code == 0 and b in ll:
                    code, method = _cell_at(*ll[b]), "gauge"
                rows.append({"Basin ID": b, "kg_code": code, "kg_frac": frac, "kg_method": method})
        new = pd.DataFrame(rows)
        cached = pd.concat([cached, new], ignore_index=True) if not cached.empty else new
        os.makedirs(os.path.dirname(cache_csv), exist_ok=True)
        cached[["Basin ID", "kg_code", "kg_frac", "kg_method"]].to_csv(cache_csv, index=False)
    out = cached[cached["Basin ID"].isin(set(wanted))][["Basin ID", "kg_code", "kg_frac", "kg_method"]].copy()
    out["kg_code"] = out["kg_code"].fillna(0).astype(int)
    out["kg_symbol"] = out["kg_code"].map(lambda c: KG_CLASSES.get(c, ("NA",))[0])
    out["kg_group"] = out["kg_code"].map(kg_group_of)
    out["kg_group_name"] = out["kg_group"].map(lambda g: KG_GROUPS[g][0] if g in KG_GROUPS else None)
    return out.reset_index(drop=True)


def koppen_summary(df_kg: pd.DataFrame) -> pd.DataFrame:
    """Basin counts per KG group and class (for quick paper tables)."""
    return (df_kg.groupby(["kg_group", "kg_group_name", "kg_symbol"]).size()
            .rename("n_basins").reset_index().sort_values(["kg_group", "kg_symbol"]))
