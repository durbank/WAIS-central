# Script for performing analyses used in article

## Set the environment

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import geoviews as gv
import holoviews as hv
from cartopy import crs as ccrs
from bokeh.io import output_notebook
output_notebook()
hv.extension('bokeh')
gv.extension('bokeh')

# Set project root directory
ROOT_DIR = Path(__file__).parents[1]

# Set project data directory
# DATA_DIR = ROOT_DIR.joinpath('data')
DATA_DIR = ROOT_DIR.joinpath('data/PAIPR-outputs')

# Import custom project functions
import sys
SRC_DIR = ROOT_DIR.joinpath('src')
sys.path.append(str(SRC_DIR))
from my_functions import *

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(
    true_scale_latitude=-71)

# Define Antarctic boundary file
shp = str(ROOT_DIR.joinpath(
    'data/Ant_basemap/Coastline_medium_res_polygon.shp'))

## Import and format PAIPR-generated results

# Import raw data
data_list = [folder for folder in DATA_DIR.glob('*')]
data_raw = pd.DataFrame()
for folder in data_list:
    data = import_PAIPR(folder)
    data_raw = data_raw.append(data)

# Remove results for below QC data reliability
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)

data_form = format_PAIPR(
    data_0, start_yr=1990, end_yr=2009).drop(
    'elev', axis=1)

# Create time series arrays for annual accumulation 
# and error
accum_ALL = data_form.pivot(
    index='Year', columns='trace_ID', values='accum')
std_ALL = data_form.pivot(
    index='Year', columns='trace_ID', values='std')

# Create gdf of mean results for each trace
data_form['collect_time'] = (
    data_form.collect_time.values.astype(np.int64))
traces = data_form.groupby('trace_ID').mean().drop(
    ['Year', 'QC_flag'], axis=1)
traces['collect_time'] = pd.to_datetime(
    traces.collect_time).dt.round('1ms')
# traces = traces.sort_values(
#     'collect_time').reset_index(drop=True)
# traces.index.name = 'trace_ID'
traces = traces.reset_index()
gdf_traces = gpd.GeoDataFrame(
    traces.drop(['Lat', 'Lon'], axis=1), 
    geometry=gpd.points_from_xy(
        traces.Lon, traces.Lat), 
    crs="EPSG:4326")
gdf_traces.to_crs(epsg=3031, inplace=True)


## Perform trend analysis

trends, _, lb, ub = trend_bs(
    accum_ALL, 1000, df_err=std_ALL)
gdf_traces['trend'] = (
    trends.values / gdf_traces['accum'])
gdf_traces['t_lb'] = lb / gdf_traces['accum']
gdf_traces['t_ub'] = ub / gdf_traces['accum']
gdf_traces['t_abs'] = trends.values


## Add factor for trend signficance

# insig_idx = gdf_traces.query(
#     't_lb<0 & t_ub>0').index.values
sig_idx = np.invert(np.array(
    [(gdf_traces['t_lb'] < 0).values, 
    (gdf_traces['t_ub'] > 0).values]).all(axis=0))


## Plot data inset map

Ant_bnds = gv.Shape.from_shapefile(
    shp, crs=ANT_proj).opts(
        projection=ANT_proj, width=500, height=500)
trace_plt = gv.Points(
    gdf_traces, crs=ANT_proj).opts(
        projection=ANT_proj, color='red')
Ant_bnds * trace_plt


## Plot of mean accumulation

accum_plt = gv.Points(
    gdf_traces, vdims=['accum', 'std'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, 
        tools=['hover'])
accum_plt.opts(width=600, height=400)


## Plot linear trend results

trend_plt = gv.Points(
    gdf_traces, vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm', symmetric=True, 
        colorbar=True, tools=['hover'])
sig_plt = gv.Points(
    gdf_traces[sig_idx], 
    vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm', symmetric=True, 
        colorbar=True, tools=['hover'])
(trend_plt.opts(width=600, height=400)
    + sig_plt.opts(width=600, height=400))


## Random time series plots

idx_i = np.random.randint(0, accum_ALL.shape[1])

t_series = accum_ALL.iloc[:,idx_i]
t_err = std_ALL.iloc[:,idx_i]

t_series.plot(color='red', linewidth=2)
(t_series+2*t_err).plot(color='red', linestyle='--')
(t_series-2*t_err).plot(color='red', linestyle='--')

## Code to aggregate results based on grids

# Useful websites for doing this...
# https://james-brennan.github.io/posts/fast_gridding_geopandas/
# https://matthewrocklin.com/blog/work/2017/09/21/accelerating-geopandas-1
# http://xarray.pydata.org/en/stable/pandas.html
# http://xarray.pydata.org/en/stable/generated/xarray.Dataset.from_dataframe.html
