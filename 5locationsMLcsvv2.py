import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import time
import concurrent.futures

# Directory and months
data_dir_2d = "/data5/wang/mesoclimate/wrf2d_d02/1998/"
months = ["199805", "199806", "199807", "199808"]

# Define stations and grid points
stations = {
    "KLBB": (301, 163),
    "KDFW": (316, 230),
    "KSAT": (256, 218),
    "KHOU": (263, 239),
    "KAMA": (324, 166),
}

# Variables to extract
vars_to_extract = [
    'T2', 'U10', 'V10', 'LPI', 'XTIME', 'GRDFLX', 'SNOW', 'SNOWH',
    'RAINC', 'RAINNC', 'I_RAINC', 'I_RAINNC', 'SNOWNC', 'GRAUPELNC',
    'HAILNC', 'OLR', 'ALBEDO', 'PBLH', 'HFX', 'QFX', 'LH',
    'W_UP_MAX', 'W_DN_MAX', 'REFD_MAX', 'UP_HELI_MAX', 'HAILCAST_DIAM_MAX',
    'IC_FLASHCOUNT', 'CG_FLASHCOUNT', 'REFD_COM', 'REFD',
    'AFWA_MSLP', 'AFWA_HEATIDX', 'AFWA_WCHILL', 'AFWA_TLYRBOT', 'AFWA_TLYRTOP',
    'AFWA_TURB', 'AFWA_LLTURB', 'AFWA_LLTURBLGT', 'AFWA_LLTURBMDT',
    'AFWA_TOTPRECIP', 'AFWA_RAIN', 'AFWA_SNOW', 'AFWA_ICE', 'AFWA_FZRA',
    'AFWA_SNOWFALL', 'AFWA_VIS', 'AFWA_VIS_DUST', 'AFWA_CLOUD', 'AFWA_CLOUD_CEIL',
    'AFWA_CAPE', 'AFWA_CIN', 'AFWA_CAPE_MU', 'AFWA_CIN_MU', 'AFWA_ZLFC',
    'AFWA_PLFC', 'AFWA_LIDX', 'AFWA_PWAT', 'AFWA_HAIL', 'AFWA_LLWS', 'AFWA_TORNADO',
    'C1H', 'C2H', 'C1F', 'C2F', 'C3H', 'C4H', 'C3F', 'C4F'
]

def process_wrf_file(file_path, station_y, station_x):
    """Opens a WRF file, extracts data for a specific grid point, and returns a DataFrame."""
    try:
        ds = xr.open_dataset(file_path)

        times_raw = ds['Times'].values
        times_str = ["".join([chr(c) for c in row]).strip() for row in times_raw]
        times = pd.to_datetime(times_str, format='%Y-%m-%d_%H:%M:%S')

        data = {"time": times}
        for var in vars_to_extract:
            if var in ds.variables:
                vals = ds[var].values
                if vals.ndim == 3:
                    data[var] = vals[:, station_y, station_x]
                elif vals.ndim == 1:
                    data[var] = vals
                else:
                    data[var] = np.full(len(times), np.nan)
            else:
                data[var] = np.full(len(times), np.nan)

        ds.close()

        return pd.DataFrame(data)

    except Exception as e:
        print(f"[Error] {file_path}: {e}")
        return None  # Skip file if there's an error

# Loop over each station
for station_name, (station_y, station_x) in stations.items():
    print(f"\n========== Processing station {station_name} at grid ({station_y},{station_x}) ==========")
    station_start = time.time()

    all_files = []
    for month in months:
        month_dir = os.path.join(data_dir_2d, month)
        file_pattern = os.path.join(month_dir, "wrf2d_d02_*")
        files_2d = sorted(glob.glob(file_pattern))
        print(f"--- Month {month}: {len(files_2d)} files found ---")
        all_files.extend(files_2d)

    # Parallel processing with ThreadPoolExecutor (safe with xarray)
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(os.cpu_count(), 8)) as executor:
        futures = [executor.submit(process_wrf_file, f, station_y, station_x) for f in all_files]

        dfs = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                dfs.append(result)

    # Concatenate all dataframes
    final_df = pd.concat(dfs, ignore_index=True)
    final_df = final_df.sort_values("time").set_index("time")

    # Save to parquet or CSV
    output_file = f"{station_name}_wrf_2d_May_Aug1998.parquet"
    final_df.to_parquet(output_file, engine="pyarrow")  # Fast output

    # For CSV instead:
    # final_df.to_csv(f"{station_name}_wrf_2d_May_Aug1998.csv")

    elapsed = time.time() - station_start
    print(f"\n✅ Completed {station_name}. Files: {len(all_files)}, Output shape: {final_df.shape}")
    print(f"Elapsed time: {elapsed:.2f} seconds")

