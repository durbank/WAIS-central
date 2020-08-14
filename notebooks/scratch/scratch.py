# This script currently is for developing time series analysis with accumulation results

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

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)

# Import Antarctic outline shapefile
ant_path = ROOT_DIR.joinpath(
    'data/Ant_basemap/Coastline_medium_res_polygon.shp')
ant_outline = gpd.read_file(ant_path)

# %%
# Import SAMBA cores
samba_raw = pd.read_excel(
    DATA_DIR.joinpath("DGK_SMB_compilation.xlsx"), 
    sheet_name='Accumulation')

core_ALL = samba_raw.iloc[3:,1:]
core_ALL.index = samba_raw.iloc[3:,0]
core_ALL.index.name = 'Year'
core_meta = samba_raw.iloc[0:3,1:]
core_meta.index = ['Lat', 'Lon', 'Elev']
core_meta.index.name = 'Attributes'
new_row = core_ALL.notna().sum()
new_row.name = 'Duration'
core_meta = core_meta.append(new_row)


core_ALL = core_ALL.transpose()
core_meta = core_meta.transpose()
core_meta.index.name = 'Name'
# %%
core_locs = gpd.GeoDataFrame(
    data=core_meta.drop(['Lat', 'Lon'], axis=1), 
    geometry=gpd.points_from_xy(
        core_meta.Lon, core_meta.Lat), 
    crs='EPSG:4326')
core_locs = core_locs.to_crs('EPSG:3031')

# %%
data_list = [dir for dir in DATA_DIR.glob('gamma/*/')]
print(f"Removed {data_list.pop(2)} from list")
print(f"Removed {data_list.pop(-1)} from list")
print(f"Removed {data_list.pop(2)} from list")
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
# Subset core accum to same time period as radar
core_accum = core_ALL.transpose()
core_accum = core_accum[
    core_accum.index.isin(
        np.arange(1979,2011))].iloc[::-1]
core_accum = core_accum.iloc[
    :,(core_accum.isna().sum() <= 0).to_list()]
core_accum.columns.name = 'Core'

# Detrend and normalize data
core_array = signal.detrend(core_accum)
core_array = (
    (core_array - core_array.mean(axis=0)) 
    / core_array.std(axis=0))

fs_core, Pxx_core = signal.welch(
    core_array, axis=0, detrend=False)
# Pxx = Pxx.astype('complex128').real

core_results = pd.DataFrame(
    Pxx_core, index=fs_core, 
    columns=core_accum.columns)

ds_core = hv.Dataset(
    (np.arange(Pxx_core.shape[1]), 
    fs_core, Pxx_core), 
    ['Core', 'Frequency'], 'Power')

plt_core = ds_core.to(
    hv.Image, ['Core', 'Frequency'])
# plt_core = ds_core.to(
#     hv.Image, ['Core', 'Frequency']).hist()
plt_core.opts(width=1200, height=800, colorbar=True)
# %%
## Calculate power specta for accum time series (both radar results and cores)

accum_tsa = accum.iloc[:,::40]

# Detrend and normalize data
accum_array = signal.detrend(accum_tsa)
accum_array = (
    (accum_array - accum_array.mean(axis=0)) 
    / accum_array.std(axis=0))

fs_accum, Pxx_accum = signal.welch(
    accum_array, axis=0, detrend=False)

accum_results = pd.DataFrame(
    Pxx_accum, index=fs_accum, 
    columns=accum_tsa.columns)


ds_accum = hv.Dataset(
    (np.arange(Pxx_accum.shape[1]), 
    fs_accum, Pxx_accum), 
    ['Trace_ID', 'Frequency'], 'Power')

plt_accum = ds_accum.to(
    hv.Image, ['Trace_ID', 'Frequency'])
plt_accum.opts(width=1200, height=800, colorbar=True)
# %%
# Compare results to white noise with the same mean and std of the detrended accum time series
ts_data = signal.detrend(accum.iloc[:,::40], axis=0)
df_test = np.random.normal(
    ts_data.mean(axis=0), ts_data.std(axis=0), 
    ts_data.shape)

df_norm = (
    (df_test - df_test.mean(axis=0)) 
    / df_test.std(axis=0))

fs_noise, Pxx_noise = signal.welch(df_norm, axis=0)

noise_results = pd.DataFrame(
    Pxx_noise, index=fs_noise, 
    columns=accum.iloc[:,::40].columns)

ds_noise = hv.Dataset(
    (np.arange(Pxx_noise.shape[1]), 
    fs_noise, Pxx_noise), 
    ['Trace_ID', 'Frequency'], 'Power')

noise_plt = ds_noise.to(
    hv.Image, ['Trace_ID', 'Frequency']).hist()
noise_plt
# %%
# 3D plot of time series analysis
import plotly.express as px
loc_tmp = core_locs.loc[core_accum.columns]
E_core = loc_tmp.geometry.x
N_core = loc_tmp.geometry.y
df_core = pd.DataFrame(
    {'Easting': E_core.repeat(Pxx_core.shape[0]), 
    'Northing': N_core.repeat(Pxx_core.shape[0]), 
    'Frequency': np.tile(fs_core, E_core.shape[0]), 
    'Power': Pxx_core.reshape(
        Pxx_core.size, 1).squeeze()})
fig = px.scatter_3d(
    df_core, 
    x='Easting', y='Northing', z='Frequency', 
    color='Power', color_continuous_scale='viridis')
fig.show()

#%%
loc_tmp = accum_trace.loc[accum_tsa.columns]
E_accum = loc_tmp.geometry.x
N_accum = loc_tmp.geometry.y
df_accum = pd.DataFrame(
    {'Easting': E_accum.repeat(Pxx_accum.shape[0]), 
    'Northing': N_accum.repeat(Pxx_accum.shape[0]), 
    'Frequency': np.tile(fs_accum, E_accum.shape[0]), 
    'Power': Pxx_accum.reshape(
        Pxx_accum.size, 1).squeeze()})

fig = px.scatter_3d(
    df_accum, 
    x='Easting', y='Northing', z='Frequency', 
    color='Power', color_continuous_scale='viridis', 
    opacity=0.75)
fig.show()

# %%
