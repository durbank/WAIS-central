# Script for performing analyses used in article

#%% Set the environment

import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
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

#%% Import and format PAIPR-generated results

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

data_form = format_PAIPR(data_0).drop(
    'elev', axis=1)
# data_form = format_PAIPR(
#     data_0, start_yr=1990, end_yr=2009).drop(
#     'elev', axis=1)

# Create time series arrays for annual accumulation 
# and error
accum_ALL = data_form.pivot(
    index='Year', columns='trace_ID', values='accum')
std_ALL = data_form.pivot(
    index='Year', columns='trace_ID', values='std')

# Create gdf of mean results for each trace and 
# transform to Antarctic Polar Stereographic
gdf_traces = long2gdf(data_form)
gdf_traces.to_crs(epsg=3031, inplace=True)



# Combine trace time series based on grid cells
tmp_grids = pts2grid(gdf_traces, resolution=2500)
gdf_grid, accum_grid, std_grid = trace_combine(
    tmp_grids, accum_ALL, std_ALL)


# Remove results that don't span at least 20 years
keep_idx = np.invert(accum_grid.isna()).sum() >= 20
gdf_grid = gdf_grid[keep_idx].reset_index(drop=True)
accum_grid = accum_grid.loc[:,keep_idx]
accum_grid.columns = gdf_grid.index
std_grid = std_grid.loc[:,keep_idx]
std_grid.columns = gdf_grid.index

#%% Perform trend analysis

trends, _, lb, ub = trend_bs(
    accum_grid, 500, df_err=std_ALL)
gdf_grid['trend'] = trends / gdf_grid['accum']
gdf_grid['t_lb'] = lb / gdf_grid['accum']
gdf_grid['t_ub'] = ub / gdf_grid['accum']
gdf_grid['t_abs'] = trends


# Add factor for trend signficance
insig_idx = gdf_grid.query(
    't_lb<0 & t_ub>0').index.values
sig_idx = np.invert(np.array(
    [(gdf_grid['t_lb'] < 0).values, 
    (gdf_grid['t_ub'] > 0).values]).all(axis=0))


#%% Plot data inset map

Ant_bnds = gv.Shape.from_shapefile(
    shp, crs=ANT_proj).opts(
        projection=ANT_proj, width=500, height=500)
trace_plt = gv.Polygons(
    gdf_grid, crs=ANT_proj).opts(
        projection=ANT_proj, line_color='red', 
        fill_color=None)
Ant_bnds * trace_plt


#%% Plot of mean accumulation

# Get 1/99 range of mean accum
# c_min = np.floor(np.quantile(gdf_traces['accum'], 0.01))
# c_max = np.ceil(np.quantile(gdf_traces['accum'], 0.99))

accum_plt = gv.Polygons(
    gdf_grid, 
    vdims=['accum', 'std'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='viridis', colorbar=True, 
        tools=['hover'])
count_plt = gv.Polygons(
    gdf_grid, vdims='num_trace', 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='magma', colorbar=True, 
        tools=['hover'])
accum_plt.opts(width=600, height=400)+count_plt.opts(width=600, height=400)

#%% Plot linear trend results

trend_plt = gv.Polygons(
    gdf_grid, vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='coolwarm', symmetric=True, 
        colorbar=True, tools=['hover'])
sig_plt = gv.Polygons(
    gdf_grid[sig_idx], 
    vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='coolwarm', symmetric=True, 
        colorbar=True, tools=['hover'])
insig_plt = gv.Polygons(
    gdf_grid.loc[insig_idx], 
    vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color_index=None, 
        fill_alpha=0.6, color='grey', 
        line_color=None, tools=['hover'])


gdf_grid['start_yr'] = (
    [accum_grid.iloc[:,idx].first_valid_index() 
    for idx in range(accum_grid.shape[1])])
yr_plt = gv.Polygons(
    gdf_grid, vdims=['start_yr', 'trend'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='plasma', colorbar=True, 
        tools=['hover'])

(sig_plt*insig_plt).opts(
    width=600, height=400) + yr_plt.opts(
        width=600, height=400)
# (trend_plt.opts(width=600, height=400)
#     + (sig_plt*insig_plt).opts(width=600, height=400))


#%% Random time series plots

idx = np.random.randint(0, accum_grid.shape[1])

t_series = accum_grid.iloc[:,idx]
t_MoE = 1.96*(
    std_grid.iloc[:,idx]
    / np.sqrt(gdf_grid['num_trace'].astype('float')[idx]))

t_series.plot(color='red', linewidth=2)
(t_series+t_MoE).plot(color='red', linestyle='--')
(t_series-t_MoE).plot(color='red', linestyle='--')


idx = [0, 200, 400, 600]
t_series = accum_grid.iloc[:,idx]
t_MoE = 1.96*(
    std_grid.iloc[:,idx]
    / np.sqrt(gdf_grid['num_trace'].astype('float')[idx]))
t_series.plot(linewidth=2)

#%%