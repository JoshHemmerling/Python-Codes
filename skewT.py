import wrf
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import numpy as np
import metpy.calc as mpcalc
from metpy.plots import SkewT
from metpy.units import units
import wrffuncs
from datetime import datetime, timedelta

"""
A standard script to make a skewT given a lat/lon point which finds 
the nearest gridbox to that point
"""

# User input for file
wrf_date_time = datetime(2024,7,10,12,00,00)
file_interval = 10
numtimeidx = 1
domain = 1
SIMULATION = "NORMAL"
lat_lon = [42.75, -73.80]

timeidx, pattern = wrffuncs.get_timeidx_and_wrf_file(wrf_date_time,  file_interval, numtimeidx,domain)
print("Using this file: ", pattern)

# Open the NetCDF file
path = f"/data4/Hemmerling/WRF-4.4.2/run/"
wrfin = Dataset(path+pattern)
 
# Convert desired coorindates to WRF gridbox coordinates
x_y = wrf.ll_to_xy(wrfin, lat_lon[0], lat_lon[1])

# Read skewT variables in
p1 = wrf.getvar(wrfin,"pressure",timeidx=timeidx)
T1 = wrf.getvar(wrfin,"tc",timeidx=timeidx)
Td1 = wrf.getvar(wrfin,"td",timeidx=timeidx)
u1 = wrf.getvar(wrfin,"ua",timeidx=timeidx)
v1 = wrf.getvar(wrfin,"va",timeidx=timeidx)

# Get variables for desired coordinates
p = p1[:,x_y[1],x_y[0]] * units.hPa
T = T1[:,x_y[1],x_y[0]] * units.degC
Td = Td1[:,x_y[1],x_y[0]] * units.degC
u = u1[:,x_y[1],x_y[0]] * units('kt')
v = v1[:,x_y[1],x_y[0]] * units('kt')

#Test if the coordinates are correct
lat1 = wrf.getvar(wrfin,"lat",timeidx=timeidx)
lon1 = wrf.getvar(wrfin,"lon",timeidx=timeidx)
lat = lat1[x_y[1],x_y[0]] * units.degree_north
lon = lon1[x_y[1],x_y[0]] * units.degree_east
print(lat)
print(lon) 

# Example of defining your own vertical barb spacing
skew = SkewT()

# Plot the data using normal plotting functions, in this case using
# log scaling in Y, as dictated by the typical meteorological plot
skew.plot(p, T, 'r')
skew.plot(p, Td, 'g')

# Set spacing interval--Every 50 mb from 1000 to 100 mb
my_interval = np.arange(100, 1000, 50) * units('mbar')

# Get indexes of values closest to defined interval
ix = mpcalc.resample_nn_1d(p, my_interval)

# Plot only values nearest to defined interval values
skew.plot_barbs(p[ix], u[ix], v[ix])

# Add the relevant special lines
skew.plot_dry_adiabats()
skew.plot_moist_adiabats()
skew.plot_mixing_lines()
skew.ax.set_ylim(1000, 100)
skew.ax.set_xlim(-60, 40)
skew.ax.set_xlabel('Temperature ($^\circ$C)')
skew.ax.set_ylabel('Pressure (hPa)')

date_format = wrf_date_time.strftime("%Y-%m-%d %H:%M:%S")
plt.title(f"SkewT at {date_format}")
plt.show()

