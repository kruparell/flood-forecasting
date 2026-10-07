import os
import sys
import time
import sqlite3
import zipfile
import io
import shutil
from pathlib import Path
from datetime import datetime
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import pandas as pd
import numpy as np
import xarray as xr
from tqdm.auto import tqdm

# Patch requests_retry_session with User-Agent and Retry backoff to handle 403 / 429
def my_retry_session(retries=5, backoff_factor=1.5, status_forcelist=(403, 429, 500, 502, 503, 504), session=None):
    s = session or requests.Session()
    s.headers.update({
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'application/json, text/html, application/xhtml+xml'
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

from rivretrieve import CanadaFetcher, PolandFetcher

def extract_canada(start_year=2016, end_year=2026):
    print("\n=======================================================")
    print(" 1. CANADA: Bulk Extraction via HYDAT SQLite Database")
    print("=======================================================")
    
    fetcher = CanadaFetcher()
    meta = CanadaFetcher.get_cached_metadata()
    print(f"Total Canadian stations in metadata: {len(meta)}")
    
    # 1. Download & get SQLite connection (Zero API calls, single file download)
    print("Connecting to / downloading HYDAT SQLite Database...")
    t0 = time.time()
    conn = fetcher._get_hydat_connection()
    t1 = time.time()
    print(f"HYDAT SQLite database ready in {t1-t0:.2f} seconds!")
    
    # 2. Query DLY_FLOWS table directly using SQL for 2016 - 2026
    print(f"Executing SQL query on DLY_FLOWS table for daily discharge across all stations ({start_year} - {end_year})...")
    sql = f"""
        SELECT STATION_NUMBER, YEAR, MONTH, 
               FLOW1, FLOW2, FLOW3, FLOW4, FLOW5, FLOW6, FLOW7, FLOW8, FLOW9, FLOW10,
               FLOW11, FLOW12, FLOW13, FLOW14, FLOW15, FLOW16, FLOW17, FLOW18, FLOW19, FLOW20,
               FLOW21, FLOW22, FLOW23, FLOW24, FLOW25, FLOW26, FLOW27, FLOW28, FLOW29, FLOW30, FLOW31
        FROM DLY_FLOWS
        WHERE YEAR >= {start_year} AND YEAR <= {end_year}
    """
    df_raw = pd.read_sql_query(sql, conn)
    print(f"Fetched {len(df_raw)} raw monthly record rows from SQL database.")
    
    # 3. Melt monthly flow records into daily timeseries DataFrame
    records = []
    print("Parsing SQL flow records into daily timeseries...")
    
    for _, row in tqdm(df_raw.iterrows(), total=len(df_raw), desc="Parsing HYDAT Records"):
        stn = row['STATION_NUMBER']
        yr = int(row['YEAR'])
        mo = int(row['MONTH'])
        
        for day in range(1, 32):
            val = row[f'FLOW{day}']
            if pd.notna(val):
                try:
                    dt = datetime(yr, mo, day)
                    records.append({'date': dt, 'basin_id': f"canada_{stn}", 'streamflow': np.float32(val)})
                except ValueError:
                    pass  # Skip invalid dates like Feb 30
                    
    df_long = pd.DataFrame(records)
    print(f"Pivoting consolidated DataFrame across {df_long['basin_id'].nunique()} Canadian stations...")
    df_pivot = df_long.pivot(index='date', columns='basin_id', values='streamflow')
    df_pivot.sort_index(inplace=True)
    
    print(f"Consolidated Canada DataFrame shape: {df_pivot.shape}")
    
    # 4. Save combined CSV and NetCDF datasets
    output_base = Path('/usr/local/google/home/kruparell/flood-forecasting/tutorial/rivretrieve/RivRetrieve_Extracted_Data/Canada/timeseries')
    csv_dir = output_base / 'csv' / 'canada'
    nc_dir = output_base / 'netcdf' / 'canada'
    csv_dir.mkdir(parents=True, exist_ok=True)
    nc_dir.mkdir(parents=True, exist_ok=True)
    
    combined_csv = output_base / 'canada_streamflow_2016_2026.csv'
    combined_nc = output_base / 'canada_streamflow_2016_2026.nc'
    
    print(f"Writing combined CSV to {combined_csv}...")
    df_pivot.to_csv(combined_csv)
    
    print(f"Writing combined NetCDF to {combined_nc}...")
    ds_canada = xr.Dataset(
        data_vars={'streamflow': (['date', 'basin_id'], df_pivot.values.astype(np.float32))},
        coords={'date': df_pivot.index.values, 'basin_id': df_pivot.columns.values},
        attrs={
            'description': 'Daily mean river discharge (m3/s) for Canadian gauges retrieved via HYDAT SQLite Database',
            'source': 'Environment Canada National Hydrometric Program (HYDAT)',
            'units': 'm3/s'
        }
    )
    ds_canada.to_netcdf(combined_nc)
    
    # 5. Export per-basin files sequentially
    saved_count = 0
    for b_id in tqdm(df_pivot.columns, desc="Exporting Canada per-basin files"):
        s_b = df_pivot[b_id].dropna()
        df_b = pd.DataFrame({'date': s_b.index.strftime('%Y-%m-%d'), 'streamflow': s_b.values})
        df_b.to_csv(csv_dir / f"{b_id}.csv", index=False)
        
        ds_b = xr.Dataset(
            data_vars={'streamflow': (['date'], s_b.values.astype(np.float32))},
            coords={'date': s_b.index.values},
            attrs={'gauge_id': b_id, 'source': 'Environment Canada HYDAT', 'units': 'm3/s'}
        )
        ds_b.to_netcdf(nc_dir / f"{b_id}.nc")
        saved_count += 1
        
    print(f"\n--- Canada Extraction Summary ---")
    print(f"Output Base Directory: {output_base}")
    print(f"Combined CSV File:    {combined_csv} ({combined_csv.stat().st_size / 1e6:.2f} MB)")
    print(f"Combined NetCDF File: {combined_nc} ({combined_nc.stat().st_size / 1e6:.2f} MB)")
    print(f"Per-Basin CSV Dir:    {csv_dir} ({saved_count} files)")
    print(f"Per-Basin NetCDF Dir: {nc_dir} ({saved_count} files)")

def main():
    print("Starting Bulk Extraction for Non-API Providers (Canada & Poland)...")
    extract_canada(2016, 2026)

if __name__ == '__main__':
    main()
