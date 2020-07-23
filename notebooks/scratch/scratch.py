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


fs, Pxx = signal.welch(
    core_accum, detrend='linear', axis=0)
Pxx = Pxx.astype('complex128').real

ds = hv.Dataset(
    (np.arange(Pxx.shape[1]), fs, Pxx), 
    ['Core', 'Frequency'], 'Power')

my_plt = ds.to(
    hv.Image, ['Core', 'Frequency']).hist()
my_plt
# %%
## Calculate power specta for accum time series (both radar results and cores)

df_test = accum.iloc[:,::40]
fs, Pxx = signal.welch(
    df_test, detrend='linear', axis=0)

df_results = pd.DataFrame(
    Pxx, index=fs, columns=df_test.columns)

ds = hv.Dataset(
    (np.arange(Pxx.shape[1]), fs, Pxx), 
    ['Trace_ID', 'Frequency'], 'Power')

my_plt = ds.to(
    hv.Image, ['Trace_ID', 'Frequency']).hist()
my_plt
# %%
