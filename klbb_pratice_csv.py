import os
import glob
import numpy as np
import pandas as pd
import xarray as xr

# Station info with grid indices
station_name = "KLBB"
station_y, station_x = 301, 163  # Use your exact WRF grid indices here

data_dir_2d = "/data3/climate/mesoclimate_wrfout/"
file_2d_pattern = "wrf2d_d02_*"
files_2d = sorted(glob.glob(os.path.join(data_dir_2d, file_2d_pattern)))

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

data_dict = {var: [] for var in vars_to_extract}
times_all = []

for f2d in files_2d:
    try:
        ds2d = xr.open_dataset(f2d)

        # Extract and decode times from 'Times' variable
        raw_times = ds2d['Times'].values
        times_str = ["".join([chr(c) for c in row]).strip() for row in raw_times]
        times = pd.to_datetime(times_str, format='%Y-%m-%d_%H:%M:%S')
        times_all.extend(times)

        for var in vars_to_extract:
            if var in ds2d.variables:
                vals = ds2d[var].values
                # Extract values at the station grid point
                if vals.ndim == 3:
                    # shape usually (Time, south_north, west_east)
                    vals_1d = vals[:, station_y, station_x]
                    data_dict[var].extend(vals_1d)
                elif vals.ndim == 1:
                    # Already 1D time series (len(times),)
                    data_dict[var].extend(vals)
                else:
                    # Unexpected shape, fill NaNs
                    data_dict[var].extend([np.nan] * len(times))
            else:
                # Variable not found in this file
                data_dict[var].extend([np.nan] * len(times))

        ds2d.close()
    except Exception as e:
        print(f"Error reading 2D file {f2d}: {e}")

# Build DataFrame with datetime index
df = pd.DataFrame(data_dict, index=pd.DatetimeIndex(times_all))

print(f"Extracted data shape: {df.shape}")
print(df.head())

# Save to CSV file named for the station
df.to_csv(f"klbb_wrf_features.csv")

