# This script currently is for investigating changes to PAIPR results after incorporating logistic parameters into MC simulations

# %%
# Import requisite modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import geoviews as gv
import holoviews as hv
from cartopy import crs as ccrs
from bokeh.io import output_notebook
from shapely.geometry import Point
output_notebook()
hv.extension('bokeh')
gv.extension('bokeh')

# Set project root directory
ROOT_DIR = Path(__file__).parent.parent.parent

# Set project data directory
DATA_DIR = ROOT_DIR.joinpath('data')

# Import custom project functions
import sys
SRC_DIR = ROOT_DIR.joinpath('src')
sys.path.append(str(SRC_DIR))
from my_functions import *

# %%
# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)

# Import Antarctic outline shapefile
ant_path = ROOT_DIR.joinpath(
    'data/Ant_basemap/Coastline_medium_res_polygon.shp')
ant_outline = gpd.read_file(ant_path)

# %%
data_list = [dir for dir in DATA_DIR.glob('Gauss/*/')]
data_raw = pd.DataFrame()
for dir in data_list:
    data = import_PAIPR(dir)
    data_raw = data_raw.append(data)

# Subset data into QC flags 0, 1, and 2
data_0 = data_raw[data_raw['QC_flag'] == 0]
data_1 = data_raw[data_raw['QC_flag'] == 1]
# data_2 = data_raw[data_raw['QC_flag'] == 2]

# Remove data_1 values earlier than assigned QC yr, 
# and recombine results with main data results
data_1 = data_1[data_1.Year >= data_1.QC_yr]
data_0 = data_0.append(data_1).sort_values(
    ['collect_time', 'Year'])

#%%
# Format accumulation data
accum_long = format_PAIPR(
    data_0, start_yr=1979, end_yr=2010).drop(
        'elev', axis=1)
traces = accum_long.groupby('trace_ID')

# New accum and std dfs in wide format
accum = accum_long.pivot(
    index='Year', columns='trace_ID', values='accum')
accum_std = accum_long.pivot(
    index='Year', columns='trace_ID', values='std')

# Create df for mean annual accumulation
accum_trace = traces.aggregate(np.mean).drop('Year', axis=1)
accum_trace = gpd.GeoDataFrame(
    accum_trace, geometry=gpd.points_from_xy(
        accum_trace.Lon, accum_trace.Lat), 
    crs="EPSG:4326").drop(['Lat', 'Lon'], axis=1)

# Convert accum crs to same as Antarctic outline
accum_trace = accum_trace.to_crs(ant_outline.crs)

# Drop original index values
accum_trace = accum_trace.drop('index', axis=1)

# %%
# Calculate robust linear trends
trends, t_intercept, t_lb, t_ub = trend_bs(
    accum, 1000)

# Add trend results to traces gdf
accum_trace['trend_abs'] = trends
accum_trace['trend_perc'] = (trends 
    / accum_trace['accum'])
accum_trace['t_lb'] = t_lb / accum_trace['accum']
accum_trace['t_ub'] = t_ub / accum_trace['accum']

# %%

tmp = pd.DataFrame(
    {'accum': accum.mean(axis=0), 
    'max_std': accum_std.max(axis=0)})
print(
    f"{((tmp.max_std/tmp.accum) > 1).sum() / len(tmp) *100:.2f}% have max errors intersecting zero")


# %%
## Plot data inset map
Ant_bnds = gv.Shape.from_shapefile(
    ant_path, crs=ANT_proj).opts(
    projection=ANT_proj, width=500, height=500)
trace_plt = gv.Points(accum_trace, crs=ANT_proj).opts(
    projection=ANT_proj, color='red')
Ant_bnds * trace_plt
# %%

accum_plt = gv.Points(
    accum_trace, 
    vdims=['accum', 'std'], 
    crs=ANT_proj).opts(projection=ANT_proj, color='accum', 
    cmap='viridis', colorbar=True, 
    tools=['hover'], width=600, height=400)
accum_plt

# %%

trend_plt = gv.Points(
    accum_trace, 
    vdims=['trend_perc', 't_lb', 't_ub'], 
    crs=ANT_proj). opts(
        projection=ANT_proj, color='trend_perc', 
        cmap='coolwarm', symmetric=True, colorbar=True, 
        tools=['hover'], width=600, height=400)
trend_plt

# %%
Tabs_plt = gv.Points(
    accum_trace, 
    vdims=['trend_abs'], 
    crs=ANT_proj). opts(
        projection=ANT_proj, color='trend_abs', 
        cmap='coolwarm', symmetric=False, colorbar=True, 
        tools=['hover'], width=600, height=400)
Tabs_plt

# %%
## Plot random accumulation time series
import matplotlib.pyplot as plt

i = np.random.randint(accum.shape[1])
yr = accum.index
smb = accum.iloc[:,i]
smb_err = accum_std.iloc[:,i]
fig, ax = plt.subplots()
ax.plot(yr, smb, color='red', lw=2)
ax.plot(yr, smb+smb_err, color='red', ls='--')
ax.plot(yr, smb-smb_err, color='red', ls='--')
fig.show()

# %%
