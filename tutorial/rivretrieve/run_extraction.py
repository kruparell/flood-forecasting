import os
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Patch requests_retry_session in rivretrieve to handle 403 / 429 retries
def my_retry_session(retries=5, backoff_factor=1.5, status_forcelist=(403, 429, 500, 502, 503, 504), session=None):
    s = session or requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json'
    })
    retry = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=status_forcelist,
        raise_on_status=False
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount('http://', adapter)
    s.mount('https://', adapter)
    return s

sys.path = [p for p in sys.path if not p.endswith('/google3')]
for p in ['/usr/local/google/home/kruparell/RivRetrieve-Python',
          '/usr/local/google/home/kruparell/flood-forecasting',
          '/usr/local/google/home/kruparell/flood-forecasting/tutorial']:
    if p not in sys.path:
        sys.path.insert(0, p)

import rivretrieve.utils
import rivretrieve.uk_ea
rivretrieve.utils.requests_retry_session = my_retry_session
rivretrieve.uk_ea.utils.requests_retry_session = my_retry_session

from rivretrieve import UKEAFetcher, constants

def main():
    caravan_attr_path = '/usr/local/google/home/kruparell/Caravans/Caravan-nc/attributes/camelsgb/attributes_caravan_camelsgb.csv'
    if not os.path.exists(caravan_attr_path):
        caravan_attr_path = '/usr/local/google/home/kruparell/flood-forecasting/tutorial/Caravan-nc/attributes/camelsgb/attributes_caravan_camelsgb.csv'

    caravan_attrs = pd.read_csv(caravan_attr_path)
    all_camelsgb_basins = caravan_attrs['gauge_id'].tolist()
    print(f"Total CAMELS-GB basins in Caravans attributes: {len(all_camelsgb_basins)}")

    fetcher = UKEAFetcher()
    meta = fetcher.get_cached_metadata()
    meta_clean = meta.copy()
    meta_clean['nrfa_clean'] = meta_clean['nrfaStationID'].dropna().astype(int, errors='ignore').astype(str)
    meta_clean['caravan_basin_id'] = 'camelsgb_' + meta_clean['nrfa_clean']

    matched_meta = meta_clean[meta_clean['caravan_basin_id'].isin(all_camelsgb_basins)].drop_duplicates(subset=['caravan_basin_id'])
    basin_to_guid = dict(zip(matched_meta['caravan_basin_id'], matched_meta['stationGuid']))
    print(f"Matched CAMELS-GB basins with UKEA station GUIDs: {len(basin_to_guid)}")

    START_DATE = '2016-01-01'
    END_DATE = '2026-12-31'

    output_base = Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/rivretrieve/RivRetrieve_Extracted_Data/CamelsGB/timeseries')
    csv_dir = output_base / 'csv' / 'camelsgb'
    nc_dir = output_base / 'netcdf' / 'camelsgb'

    csv_dir.mkdir(parents=True, exist_ok=True)
    nc_dir.mkdir(parents=True, exist_ok=True)

    def fetch_one(b_id):
        guid = basin_to_guid.get(b_id)
        if not guid:
            return b_id, None
        try:
            df = fetcher.get_data(guid, constants.DISCHARGE_DAILY_MEAN, START_DATE, END_DATE)
            if df.empty or 'discharge_daily_mean' not in df.columns:
                return b_id, None
            s = df['discharge_daily_mean']
            s_idx = pd.to_datetime(s.index).tz_localize(None).normalize()
            s_series = pd.Series(data=s.values.astype(np.float32), index=s_idx)
            s_series = s_series[~s_series.index.duplicated(keep='first')]
            s_series.name = b_id
            return b_id, s_series
        except Exception:
            return b_id, None

    print(f"Extracting river discharge data from {START_DATE} to {END_DATE}...")
    t0 = time.time()
    series_dict = {}

    with ThreadPoolExecutor(max_workers=30) as executor:
        futures = [executor.submit(fetch_one, b_id) for b_id in all_camelsgb_basins]
        for future in tqdm(as_completed(futures), total=len(all_camelsgb_basins), desc="Fetching CAMELS-GB Basins"):
            b_id, s_series = future.result()
            if s_series is not None:
                series_dict[b_id] = s_series

    t1 = time.time()
    print(f"Extraction complete! Successfully retrieved streamflow for {len(series_dict)} basins in {t1-t0:.2f} seconds.")

    df_discharge = pd.DataFrame(series_dict)
    df_discharge.index.name = 'date'
    df_discharge.sort_index(inplace=True)
    df_discharge.dropna(how='all', inplace=True)
    print(f"Consolidated DataFrame shape: {df_discharge.shape}")

    combined_csv_path = output_base / 'camelsgb_streamflow_2016_2026.csv'
    combined_nc_path = output_base / 'camelsgb_streamflow_2016_2026.nc'

    df_discharge.to_csv(combined_csv_path)

    ds_discharge = xr.Dataset(
        data_vars={
            'streamflow': (['date', 'basin_id'], df_discharge.values.astype(np.float32))
        },
        coords={
            'date': df_discharge.index.values,
            'basin_id': df_discharge.columns.values
        },
        attrs={
            'description': 'Daily mean river discharge (m3/s) for CAMELS-GB basins retrieved via RivRetrieve (UKEA API)',
            'source': 'UK Environment Agency (UKEA) Hydrology API via RivRetrieve',
            'time_period': f"{df_discharge.index.min().strftime('%Y-%m-%d')} to {df_discharge.index.max().strftime('%Y-%m-%d')}",
            'units': 'm3/s'
        }
    )
    ds_discharge.to_netcdf(combined_nc_path)

    saved_count = 0
    for b_id in tqdm(df_discharge.columns, desc="Exporting per-basin files"):
        s_b = df_discharge[b_id].dropna()
        df_b = pd.DataFrame({'date': s_b.index.strftime('%Y-%m-%d'), 'streamflow': s_b.values})
        df_b.to_csv(csv_dir / f"{b_id}.csv", index=False)
        ds_b = xr.Dataset(
            data_vars={'streamflow': (['date'], s_b.values.astype(np.float32))},
            coords={'date': s_b.index.values},
            attrs={'gauge_id': b_id, 'source': 'UK Environment Agency (UKEA) Hydrology API via RivRetrieve', 'units': 'm3/s'}
        )
        ds_b.to_netcdf(nc_dir / f"{b_id}.nc")
        saved_count += 1

    print(f"\n--- Extracted Dataset Summary ---")
    print(f"Output Base Directory: {output_base}")
    print(f"Combined CSV File:    {combined_csv_path} ({combined_csv_path.stat().st_size / 1e6:.2f} MB)")
    print(f"Combined NetCDF File: {combined_nc_path} ({combined_nc_path.stat().st_size / 1e6:.2f} MB)")
    print(f"Per-Basin CSV Dir:    {csv_dir} ({saved_count} files)")
    print(f"Per-Basin NetCDF Dir: {nc_dir} ({saved_count} files)")

    # Plot Hydrographs
    sample_basins = ['camelsgb_45001', 'camelsgb_24003', 'camelsgb_25020', 'camelsgb_101002']
    sample_basins = [b for b in sample_basins if b in df_discharge.columns]
    if len(sample_basins) < 4:
        sample_basins = df_discharge.columns[:4].tolist()

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), dpi=120)
    axes = axes.flatten()

    for i, b_id in enumerate(sample_basins):
        s = df_discharge[b_id].dropna()
        axes[i].plot(s.index, s.values, label=f'{b_id} Discharge', color='navy', lw=1.2)
        axes[i].set_title(f'Hydrograph: {b_id} (2016 - 2026)', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Streamflow ($m^3/s$)', fontsize=10)
        axes[i].set_xlabel('Date', fontsize=10)
        axes[i].grid(True, linestyle='--', alpha=0.5)
        axes[i].legend(loc='upper right')

    plt.suptitle('CAMELS-GB Extracted Daily River Discharge Hydrographs via RivRetrieve', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plot_path = output_base.parent / 'camelsgb_hydrographs_sample.png'
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved hydrograph plot to {plot_path}")

if __name__ == '__main__':
    main()
