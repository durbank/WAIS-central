# Script for performing analyses used in article

# %% Set the environment

# Import modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import geoviews as gv
import holoviews as hv
from cartopy import crs as ccrs
from bokeh.io import output_notebook
output_notebook()
hv.extension('bokeh')
gv.extension('bokeh')
from shapely.geometry import Polygon, Point

# hv.archive.auto()

# Set project root directory
ROOT_DIR = Path(__file__).parents[1]

# Set project data directory
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
gdf_ANT = gpd.read_file(shp)


import xarray as xr
# Define Antarctic DEM file
xr_DEM = xr.open_rasterio(
    ROOT_DIR.joinpath(
        'data/Antarctica_Cryosat2_1km_DEMv1.0.tif')).squeeze()

#%% Import and format PAIPR-generated results

# Import raw data
data_list = [folder for folder in DATA_DIR.glob('*')]
# data_list = [
#     path for path in data_list 
#     if "20141103" not in str(path)] #Removes 2014 results, as there are too few and are suspect
data_raw = pd.DataFrame()
for folder in data_list:
    data = import_PAIPR(folder)
    data_raw = data_raw.append(data)

# Remove results for below QC data reliability
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)

# Format and sort results for further processing
# data_form = format_PAIPR(data_0).drop(
#     'elev', axis=1)
data_form = format_PAIPR(data_0)

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

# %% Import and format SAMBA cores

# Import raw data
samba_raw = pd.read_excel(ROOT_DIR.joinpath(
    "data/DGK_SMB_compilation.xlsx"), 
    sheet_name='Accumulation')

# Format SAMBA core data
core_ALL = samba_raw.iloc[3:,1:]
core_ALL.index = samba_raw.iloc[3:,0]
core_ALL.index.name = 'Year'
core_meta = samba_raw.iloc[0:3,1:]
core_meta.index = ['Lat', 'Lon', 'Elev']
core_meta.index.name = 'Attributes'
new_row = core_ALL.notna().sum()
new_row.name = 'Duration'
core_meta = core_meta.append(new_row)

# Flip to get results in tidy-compliant form
# core_ALL = core_ALL.transpose()
core_meta = core_meta.transpose()
core_meta.index.name = 'Name'

# Convert to geodf and reproject to 3031
core_locs = gpd.GeoDataFrame(
    data=core_meta.drop(['Lat', 'Lon'], axis=1), 
    geometry=gpd.points_from_xy(
        core_meta.Lon, core_meta.Lat), 
    crs='EPSG:4326')
core_locs.to_crs('EPSG:3031', inplace=True)

# Define core bounding box
bbox = Polygon(
    [[-2.41E6,1.53E6], [-2.41E6,-7.78E5],
    [-4.70E5,-7.78E5], [-4.70E5,1.53E6]])

# Subset core results to region of interest
keep_idx = core_locs.within(bbox)
gdf_cores = core_locs[keep_idx]
core_ACCUM = core_ALL.loc[:,keep_idx].sort_index()

# Remove cores with less than 5 years of data
gdf_cores.query('Duration >= 10', inplace=True)

# Remove cores with missing elev data
# (this only gets rid of Ronne ice shelf cores)
gdf_cores = gdf_cores[gdf_cores['Elev'].notna()]

# Remove specific unwanted cores
gdf_cores.drop('SEAT-10-4', inplace=True)
gdf_cores.drop('BER11C95_25', inplace=True)
gdf_cores.drop('SEAT-11-1', inplace=True)
gdf_cores.drop('SEAT-11-2', inplace=True)
gdf_cores.drop('SEAT-11-3', inplace=True)
gdf_cores.drop('SEAT-11-4', inplace=True)
gdf_cores.drop('SEAT-11-6', inplace=True)
gdf_cores.drop('SEAT-11-7', inplace=True)
gdf_cores.drop('SEAT-11-8', inplace=True)

# Remove additional cores from core time series
core_ACCUM = core_ACCUM[gdf_cores.index]

# %%

# Combine trace time series based on grid cells
tmp_grids = pts2grid(gdf_traces, resolution=2500)
(gdf_grid_ALL, accum_grid_ALL, 
    MoE_grid_ALL, yr_count_ALL) = trace_combine(
    tmp_grids, accum_ALL, std_ALL)

# Subset results to period 1979-2010
yr_start = 1979
yr_end = 2009
keep_idx = np.invert(
    accum_grid_ALL.loc[yr_start:yr_end,:].isnull().any())
accum_grid = accum_grid_ALL.loc[yr_start:yr_end,keep_idx]
MoE_grid = MoE_grid_ALL.loc[yr_start:yr_end,keep_idx]
gdf_grid = gdf_grid_ALL[keep_idx]
gdf_grid['accum'] = accum_grid.mean()
gdf_grid['MoE'] = MoE_grid.mean()

# Subset cores to same time period
keep_idx = np.invert(
    core_ACCUM.loc[yr_start:yr_end,:].isnull().any())
accum_core = core_ACCUM.loc[yr_start:yr_end,keep_idx]
gdf_core = gdf_cores[keep_idx]
gdf_core['accum'] = accum_core.mean()

# %% Calculate sig/insig linear trends

# Calculate trends in radar
trends, _, lb, ub = trend_bs(
    accum_grid, 1000, df_err=MoE_grid)
gdf_grid['trend'] = trends
gdf_grid['t_lb'] = lb
gdf_grid['t_ub'] = ub
gdf_grid['t_perc'] = trends / gdf_grid['accum']
# gdf_grid['trend'] = trends / gdf_grid['accum']
# gdf_grid['t_lb'] = lb / gdf_grid['accum']
# gdf_grid['t_ub'] = ub / gdf_grid['accum']
# gdf_grid['t_abs'] = trends

# Add factor for trend signficance
insig_idx = gdf_grid.query(
    't_lb<0 & t_ub>0').index.values
sig_idx = np.invert(np.array(
    [(gdf_grid['t_lb'] < 0).values, 
    (gdf_grid['t_ub'] > 0).values]).all(axis=0))

# Calculate trends in cores
trends, _, lb, ub = trend_bs(accum_core, 1000)
gdf_core['trend'] = trends
gdf_core['t_lb'] = lb
gdf_core['t_ub'] = ub
gdf_core['t_perc'] = trends / gdf_core['accum']

# %% Calculate trends using non-Bootstrapping

import statsmodels.api as sm

rlm_param = []
rlm_Tlb = []
rlm_Tub = []

for name, series in accum_grid.items():
    X = sm.add_constant(series.index.values)
    y = series.values
    # W = 1/(std_grid.loc[:,name].values**2)
    # mod = sm.WLS(y,X, weights=W).fit()
    mod = sm.RLM(y,X).fit()
    rlm_param.append(mod.params[1])
    # wls_r2.append(mod.rsquared)
    rlm_Tlb.append(mod.conf_int()[1,0])
    rlm_Tub.append(mod.conf_int()[1,1])

gdf_grid['rlm_T'] = rlm_param
gdf_grid['rlm_lb'] = rlm_Tlb
gdf_grid['rlm_ub'] = rlm_Tub


# %% Calculate trends for partial-coverage cores

# Keep all core data from start of
# selected radar data to present
cores_long = core_ACCUM.loc[yr_start:]
gdf_long = gdf_cores
gdf_long['accum'] = cores_long.mean()

# Create size variable based on number of years in record
tmp = np.invert(
    cores_long.isna()).sum()
gdf_long['size'] = 20*tmp/tmp.max()

# Calculate trends in cores
trends, _, lb, ub = trend_bs(cores_long, 1000)
gdf_long['trend'] = trends
gdf_long['t_lb'] = lb
gdf_long['t_ub'] = ub
gdf_long['t_perc'] = trends / gdf_long['accum']

# Determine trend significance for cores
insig_core = gdf_long.query(
    't_lb<0 & t_ub>0').index.values
sig_core = np.invert(np.array(
    [(gdf_long['t_lb'] < 0).values, 
    (gdf_long['t_ub'] > 0).values]).all(axis=0))

# %% Study site inset plot

gdf_bounds = {
    'x_range': tuple(gdf_grid.total_bounds[0::2]+[-25000,25000]), 
    'y_range': tuple(gdf_grid.total_bounds[1::2]+[-30000,25000])}

# Bounds of radar data
poly_bnds = Polygon([
    [gdf_bounds['x_range'][0], gdf_bounds['y_range'][0]], 
    [gdf_bounds['x_range'][0], gdf_bounds['y_range'][1]], 
    [gdf_bounds['x_range'][1], gdf_bounds['y_range'][1]], 
    [gdf_bounds['x_range'][1], gdf_bounds['y_range'][0]]
    ])

radar_bnds = gv.Polygons(
    poly_bnds, crs=ANT_proj).opts(
        projection=ANT_proj, line_color='red', 
        line_width=3, color=None)

# Antarctica boundaries
Ant_bnds = gv.Shape.from_shapefile(
    shp, crs=ANT_proj).opts(
        projection=ANT_proj, color='silver')

# Add plot to workspace
(Ant_bnds * radar_bnds).opts(
    width=700, height=700, bgcolor='lightsteelblue')

# %%

# Clip elevation data to radar bounds
xr_DEM = xr_DEM.sel(
    x=slice(poly_bnds.bounds[0], poly_bnds.bounds[2]), 
    y=slice(poly_bnds.bounds[3],poly_bnds.bounds[1]))

# Generate elevation contours plot
elev_plt = hv.Image(xr_DEM.values, bounds=poly_bnds.bounds)
cont_plt = hv.operation.contours(elev_plt, levels=15)

# %% Plot mean accumulation

ac_max = gdf_grid.accum.max()
ac_min = gdf_grid.accum.min()
# ac_max = np.max(
#     [gdf_core.accum.max(), gdf_grid.accum.max()])
# ac_min = np.min(
#     [gdf_core.accum.min(), gdf_grid.accum.min()])

accum_plt = gv.Polygons(
    gdf_grid, 
    vdims=['accum', 'MoE'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='viridis', colorbar=True, 
        tools=['hover'])
accum_core_plt = gv.Points(
    gdf_long, vdims=['Name', 'accum', 'size'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, 
        line_color='black', size='size', 
        marker='triangle', tools=['hover'])
count_plt = gv.Polygons(
    gdf_grid, vdims='trace_count', 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='magma', colorbar=True, 
        tools=['hover'])

# %% Plot linear trend results

t_max = np.max(
    [gdf_grid.trend.max(), gdf_core.trend.max()])
t_min = np.min(
    [gdf_grid.trend.min(), gdf_core.trend.min()])

gdf_ERR = gdf_grid.copy().drop(
    ['accum', 'MoE', 't_lb', 
    't_ub', 't_perc'], axis=1)
gdf_ERR['t_MoE'] = (
    gdf_grid['t_ub'] 
    - gdf_grid['t_lb']) / 2

sig_plt = gv.Polygons(
    gdf_grid[sig_idx], 
    vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, tools=['hover'])
insig_plt = gv.Polygons(
    gdf_grid.loc[insig_idx], 
    vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color_index=None, 
        fill_alpha=0.75, color='grey', 
        line_color=None, tools=['hover'])
sig_core_plt = gv.Points(
    gdf_long[sig_core], 
    vdims=['Name', 'trend', 't_lb', 't_ub', 'size'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, size='size', line_color='black', 
        marker='triangle', tools=['hover'])
insig_core_plt = gv.Points(
    gdf_long.loc[insig_core], 
    vdims=['Name', 'trend', 't_lb', 't_ub', 'size'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='grey', alpha=0.75,  
        size='size', line_color='black', 
        marker='triangle', tools=['hover'])
tERR_plt = gv.Polygons(
    gdf_ERR, vdims=['t_MoE', 'trend'], 
    crs=ANT_proj).opts(projection=ANT_proj, 
    line_color=None, cmap='plasma', colorbar=True, 
    tools=['hover'])


t_plt = gv.Polygons(
    data=gdf_grid, 
    vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, tools=['hover'])
core_t_plt = gv.Points(
    data=gdf_long, 
    vdims=['Name', 'trend', 't_lb', 't_ub', 'size'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, size='size', line_color='black', 
        marker='triangle', tools=['hover'])


sig_pltALL = (
    insig_plt*sig_plt*insig_core_plt
    *sig_core_plt.redim.range(trend=(t_min,t_max))
)

(
    (elev_plt.opts(cmap='dimgray')
        # * cont_plt.opts(cmap='dimgray')
        * t_plt*core_t_plt.redim.range(trend=(-15,15))
    ).opts(
        width=700, height=700, 
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range']) 
    + (elev_plt.opts(cmap='dimgray') * tERR_plt).opts(
        width=700, height=700, 
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])
    + (elev_plt.opts(cmap='dimgray') * sig_pltALL).opts(
        width=700, height=700, 
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])
)
#     + (elev_plt.opts(cmap='dimgray') 
#         * count_plt.redim.range(trace_count=(0,30))).opts(
#             width=600, height=400)
# )




# %% Trends with all cores (not all cover full time period)

# Plot trends since 1979 (all)
# t_max = np.max(
#     [gdf_grid.trend.max(), gdf_long.trend.max()])
# t_min = np.min(
#     [gdf_grid.trend.min(), gdf_long.trend.min()])

core_bounds = {
    'x_range': bbox.bounds[0::2], 
    'y_range': bbox.bounds[1::2]}

rLOC_plt = gv.Polygons(
    data=gdf_grid, crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        color='black')

coreALL_plt = gv.Points(
    data=gdf_long, 
    vdims=['Name', 'trend', 't_lb', 't_ub', 'size'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, size='size', 
        line_color='black', 
        marker='triangle', tools=['hover'])

(
    coreALL_plt*Ant_bnds*rLOC_plt*coreALL_plt.opts(
        bgcolor='lightsteelblue').redim.range(trend=(-15,15))
).opts(
    xlim=core_bounds['x_range'], 
    ylim=core_bounds['y_range'], 
    width=700, height=700)

# %% Limit time series to those within the bounds

# trace_idx = gdf_traces.within(poly_bnds)
# gdf_traces = gdf_traces[trace_idx]
# accum_ALL = 

poly_idx = gdf_grid_ALL.within(poly_bnds)
gdf_grid_ALL = gdf_grid_ALL[poly_idx]
accum_grid_ALL = accum_grid_ALL[gdf_grid_ALL.index]
MoE_grid_ALL = MoE_grid_ALL[gdf_grid_ALL.index]
yr_count_ALL = yr_count_ALL[gdf_grid_ALL.index]

# %% Mean accum and trends for full time series

# Limit full time series to those with 10+ years of records
gdf_grid_ALL['Duration'] = accum_grid_ALL.notna().sum()
gdf_grid_ALL.query('Duration >= 10', inplace=True)
accum_grid_ALL = accum_grid_ALL[gdf_grid_ALL.index]
MoE_grid_ALL = MoE_grid_ALL[gdf_grid_ALL.index]
yr_count_ALL = yr_count_ALL[gdf_grid_ALL.index]

# Calculate limits for colormap
ac_max = np.quantile(gdf_grid_ALL.accum, 0.99)
ac_min = np.quantile(gdf_grid_ALL.accum, 0.01)

# Plot of mean accumulation across full time
aALL_plt = gv.Polygons(
    gdf_grid_ALL, vdims=['accum', 'MoE'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, 
        line_color=None, tools=['hover'])

# Plot of time series durations
duration_plt = gv.Polygons(
    gdf_grid_ALL, vdims=['Duration', 'trace_count'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='Duration', 
        cmap='magma', colorbar=True, 
        line_color=None, tools=['hover'])

(
    (elev_plt.opts(cmap='dimgray')
    * (aALL_plt*accum_core_plt).redim.range(
        accum=(ac_min,ac_max))
    ).opts(
        width=700, height=700,  
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])
    + (elev_plt.opts(cmap='dimgray')*duration_plt).opts(
        width=700, height=700, 
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])
)

# %% 1979-2010 accum plot

(
    elev_plt.opts(cmap='dimgray', colorbar=False)
    * (accum_plt*accum_core_plt).redim.range(
        accum=(250,500))
).opts(
    width=700, height=700, 
    xlim=gdf_bounds['x_range'], 
    ylim=gdf_bounds['y_range'])

# %%

# Calculate % change between mean accum 1979-2010 
# and mean accum for max duration available
ALL_subset = gdf_grid_ALL.loc[
    gdf_grid_ALL['grid_ID'].isin(gdf_grid['grid_ID'])]
res_accum = (ALL_subset['accum'] - gdf_grid['accum'])
rmse_accum = np.sqrt(
    (res_accum**2).sum() / (res_accum.count()-1))
print(f"Mean bias between 1979-2010 mean accum and mean accum for full available duration is {res_accum.mean():.2f} mm/yr ({100*(res_accum/gdf_grid['accum']).mean():.2f}%)")
print(f"RMSE between 1979-2010 mean accum and mean for full duration is {rmse_accum:.2f} mm/yr ({100*rmse_accum/gdf_grid['accum'].mean():.2f}%)")

res_gdf = gpd.GeoDataFrame(
    data=100*res_accum/gdf_grid.accum, 
    geometry=ALL_subset.geometry, 
    crs=ALL_subset.crs)

res_plt = gv.Polygons(
    data=res_gdf, vdims=['accum'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='accum', 
        cmap='PRGn', colorbar=True, symmetric=True,
        line_color=None, tools=['hover'])

res_min = np.quantile(res_gdf.accum, 0.01)
res_max = np.quantile(res_gdf.accum, 0.99)

(
    elev_plt.opts(cmap='dimgray', colorbar=False)
    * res_plt.redim.range(accum=(res_min,res_max))
    ).opts(
        width=700, height=700,  
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])

# %% Data location map

# Radar location plot
radar_plt = gv.Polygons(
    gdf_grid_ALL, crs=ANT_proj).opts(
        projection=ANT_proj, color='red', line_color=None)

# Core location plot
core_plt = gv.Points(
    gdf_long, crs=ANT_proj, 
    vdims=['Name']).opts(
        projection=ANT_proj, color='blue', size=15, 
        line_color='black', marker='triangle', 
        tools=['hover'])

# Add to workspace
(
    elev_plt.opts(cmap='dimgray', colorbar=True) 
    * radar_plt * core_plt
    ).opts(
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'], 
        width=700, height=700)

# %% 

# # Calculate trends for full duration
# trends, _, lb, ub = trend_bs(
#     accum_grid_ALL, 1000, df_err=std_grid_ALL)
# gdf_grid_ALL['trend'] = trends
# gdf_grid_ALL['t_lb'] = lb
# gdf_grid_ALL['t_ub'] = ub
# gdf_grid_ALL['t_perc'] = trends / gdf_grid_ALL['accum']

# %%

# tr_max = np.quantile(gdf_grid_ALL.trend, 0.99)
# tr_min = np.quantile(gdf_grid_ALL.trend, 0.01)
# tALL_plt = gv.Polygons(
#     data=gdf_grid_ALL, 
#     vdims=['trend', 't_lb', 't_ub'], 
#     crs=ANT_proj).opts(
#         projection=ANT_proj, line_color=None, 
#         cmap='coolwarm_r', symmetric=True, 
#         colorbar=True, tools=['hover'])


# gdf_ERR_ALL = gdf_grid_ALL.copy().drop(
#     ['accum', 'std', 't_lb', 
#     't_ub', 't_perc'], axis=1)
# gdf_ERR_ALL['MoE'] = (
#     gdf_grid_ALL['t_ub'] 
#     - gdf_grid_ALL['t_lb']) / 2
# ERR_max = np.quantile(gdf_ERR_ALL.MoE, 0.99)
# ERR_min = np.quantile(gdf_ERR_ALL.MoE, 0.01)

# ERR_all_plt = gv.Polygons(
#     gdf_ERR_ALL, vdims=['MoE', 'trend'], 
#     crs=ANT_proj).opts(projection=ANT_proj, 
#     line_color=None, cmap='plasma', colorbar=True, 
#     tools=['hover'])

# (
#     tALL_plt.opts(
#         width=600, height=400, 
#         bgcolor='silver').redim.range(
#         trend=(tr_min,tr_max))
#     + ERR_all_plt.opts(
#         width=600, height=400, 
#         bgcolor='silver').redim.range(
#             MoE=(ERR_min,ERR_max))
# )


# %% Larger grid cells and composite time series

cores_accum1960 = core_ACCUM.loc[1960::]
gdf_cores['accum'] = core_ACCUM.loc[
    1960::,gdf_cores.index].mean()


# Combine trace time series based on grid cells
tmp_grid_big = pts2grid(gdf_traces, resolution=100000)
(gdf_BIG, accum_BIG, 
    MoE_BIG, yr_count_BIG) = trace_combine(
    tmp_grid_big, accum_ALL, std_ALL)

# Limit time series to those within the bounds
poly_idx = gdf_BIG.intersects(poly_bnds)
gdf_BIG = gdf_BIG[poly_idx]
accum_BIG = accum_BIG[gdf_BIG.index]
MoE_BIG = MoE_BIG[gdf_BIG.index]
yr_count_BIG = yr_count_BIG[gdf_BIG.index]

# Add the total duration of the time series for each grid
gdf_BIG['Duration'] = accum_BIG.notna().sum()

# Calculate trends for large grid cells
trends, _, lb, ub = trend_bs(
    accum_BIG, 1000, df_err=MoE_BIG)
gdf_BIG['trend'] = trends
gdf_BIG['t_lb'] = lb
gdf_BIG['t_ub'] = ub
gdf_BIG['t_perc'] = trends / gdf_BIG['accum']

# %%

# Set colormap limits for trends
tr_max = np.quantile(gdf_BIG.trend, 0.99)
tr_min = np.quantile(gdf_BIG.trend, 0.01)

# Trend map for big grids
tBIG_plt = gv.Polygons(
    data=gdf_BIG, 
    vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, tools=['hover'])

# Calculate trend significance for big grids
sig_BIG = np.invert(np.array(
    [(gdf_BIG['t_lb'] < 0).values, 
    (gdf_BIG['t_ub'] > 0).values]).all(axis=0))

# Generate polygons for signficant trends
sig_plt = gv.Polygons(
    data=gdf_BIG[sig_BIG], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color='black', 
        line_width=2, color=None)

# Radar data plot
radar_plt = gv.Polygons(
    gdf_grid_ALL, crs=ANT_proj).opts(
        projection=ANT_proj, 
        line_color=None, color='black')

# Calculate colormap limits for trend margin of error
ERR_BIG = gdf_BIG.copy().drop(
    ['accum', 'MoE', 't_lb', 
    't_ub', 't_perc'], axis=1)
ERR_BIG['t_MoE'] = (
    gdf_BIG['t_ub'] 
    - gdf_BIG['t_lb']) / 2
ERR_max = np.quantile(ERR_BIG.t_MoE, 0.99)
ERR_min = np.quantile(ERR_BIG.t_MoE, 0.01)

# Margin of error on trends plot
ERR_BIG_plt = gv.Polygons(
    ERR_BIG, vdims=['t_MoE', 'trend'], 
    crs=ANT_proj).opts(projection=ANT_proj, 
    line_color=None, cmap='plasma', colorbar=True, 
    tools=['hover'])

(
    (tBIG_plt*sig_plt*radar_plt).opts(
        width=700, height=700, 
        bgcolor='silver').redim.range(
        trend=(tr_min,tr_max))
    + (ERR_BIG_plt.redim.range(MoE=(ERR_min,ERR_max))
        * radar_plt).opts(width=700, height=700, bgcolor='silver')
)

# %% Time series correlations to create clustered groups

# gdf_tmp = gdf_BIG.copy()
# corr_tmp = accum_BIG.copy()
# groups = []

# while gdf_tmp.shape[0] > 1:
#     corr_BIG = corr_tmp.diff().corr()
#     col_idx = corr_BIG[corr_BIG > 0].sum().idxmax()
#     group_idx = corr_BIG[corr_BIG[col_idx] >= 0.50].index
#     groups.append(group_idx.values)

#     gdf_tmp.drop(group_idx, inplace=True)
#     corr_tmp.drop(group_idx, axis=1, inplace=True)

# %% K-means clustering to create groups

from sklearn.cluster import KMeans

norm_df = (
    ((accum_BIG-accum_BIG.mean())
    / accum_BIG.std()).T).loc[:,1979:2010]
# norm_df = (accum_BIG.T)
norm_df['East'] = (
    gdf_BIG.centroid.x-gdf_BIG.centroid.x.mean()) \
    / gdf_BIG.centroid.x.std()
norm_df['North'] = (
    gdf_BIG.centroid.y-gdf_BIG.centroid.y.mean()) \
    / gdf_BIG.centroid.y.std()


norm_df[norm_df.isna()] = 0

grp_pred = KMeans(
    n_clusters=4, 
    # n_clusters=3, 
    random_state=0).fit_predict(norm_df)
gdf_BIG['Group'] = grp_pred
alpha_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
gdf_BIG.replace({'Group': alpha_dict}, inplace=True)

# %%

grp_A = [22, 23, 29, 35]
grp_B = [27, 28, 33, 34, 39, 40]
grp_C = [37, 38, 43, 49]
grp_D = [19, 20, 21, 25, 26, 31, 32]

gdf_BIG['Group'] = ""

for i, group in enumerate(
    [grp_A, grp_B, grp_C, grp_D]):

    row_idx = gdf_BIG['grid_ID'].isin(group)
    gdf_BIG.loc[row_idx,'Group'] = i

alpha_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
gdf_BIG.replace({'Group': alpha_dict}, inplace=True)

# %%

# poly_groups = []
# G_cmap = {'A':'#66C2A5', 'B':'#FC8D62', 'C':'#8DA0CB', 'D':'#E78AC3'}

# for i, group in enumerate(
#     gdf_BIG.groupby('Group').groups):

#     # Subset data based on group identity
#     gdf_group = gdf_BIG[gdf_BIG['Group'] == group]
#     accum_group = accum_BIG[gdf_group.index]
#     count_group = yr_count_BIG[gdf_group.index]
    
#     # Merge group into single polygon
#     poly = gdf_group.geometry.unary_union
#     poly_groups.append(poly)

#     # Build composite core from all cores withing poly
#     cores_group = gdf_cores[
#         gdf_cores.geometry.within(poly)]
#     cores_Gaccum = core_ACCUM[cores_group.index].loc[
#             accum_group.index[0]:]
#     core_comp = cores_Gaccum.mean(axis=1)
#     core_count = cores_Gaccum.notna().sum(axis=1)
    
#     fig, (ax1, ax2) = plt.subplots(2, 1)
#     fig.suptitle(
#         'Grid Group '+alpha_dict[i]+' time series')
#     accum_group.plot(ax=ax1, color=G_cmap[group], 
#     label='_hidden_')
#     core_comp.plot(
#         ax=ax1, color='grey', linewidth=2, 
#         linestyle='--')
#     count_group.plot(ax=ax2, color=G_cmap[group])
#     ax1.get_legend().remove()
#     ax1.set_xlim(
#         [accum_BIG.index[0], 
#         accum_BIG.index[-1]])
#     ax1.set_ylabel('SMB (mm/a)')
#     ax2.get_legend().remove()
#     ax2.set_xlim(
#         [accum_BIG.index[0], 
#         accum_BIG.index[-1]])
#     ax2.set_ylabel('# traces')

#     if core_count.sum():
#         ax3=ax2.twinx()
#         core_count.plot(
#             ax=ax3, color='grey', linewidth=2, 
#             linestyle='--')
#         ax3.set_ylabel('# Cores')
#     plt.show()

# %%

poly_groups = []
G_cmap = {
    'A':'#66C2A5', 'B':'#FC8D62', 
    'C':'#8DA0CB', 'D':'#E78AC3'}


fig = plt.figure(figsize=(13, 10))
outer = gridspec.GridSpec(
    2, 2, wspace=0.4, hspace=0.3)

for i, group in enumerate(
    gdf_BIG.groupby('Group').groups):

    # Subset data based on group identity
    gdf_group = gdf_BIG[gdf_BIG['Group'] == group]
    accum_group = accum_BIG[gdf_group.index]
    count_group = yr_count_BIG[gdf_group.index]
    
    # Merge group into single polygon
    poly = gdf_group.geometry.unary_union
    poly_groups.append(poly)

    # Build composite core from all cores within poly
    cores_group = gdf_cores[
        gdf_cores.geometry.within(poly)]
    cores_Gaccum = core_ACCUM[cores_group.index].loc[
            accum_group.index[0]:]
    core_comp = cores_Gaccum.mean(axis=1)
    core_count = cores_Gaccum.notna().sum(axis=1)
    

    inner = gridspec.GridSpecFromSubplotSpec(
        2, 1,subplot_spec=outer[i], 
        wspace=0.2, hspace=0.25)
    ax1 = plt.Subplot(fig, inner[0])
    ax2 = plt.Subplot(fig, inner[1])
    ax1.set_title(
        'Grid Group '+alpha_dict[i]+' time series')

    # fig.suptitle(
    #     'Grid Group '+alpha_dict[i]+' time series')

    accum_group.plot(ax=ax1, color=G_cmap[group], 
    label='_hidden_')
    core_comp.plot(
        ax=ax1, color='grey', linewidth=2, 
        linestyle='--')
    count_group.plot(ax=ax2, color=G_cmap[group])
    ax1.get_legend().remove()
    ax1.set_xlim(
        [accum_BIG.index[0], 
        accum_BIG.index[-1]])
    ax1.set_xlabel(None)
    ax1.set_ylabel('SMB (mm/a)')
    ax2.get_legend().remove()
    ax2.set_xlim(
        [accum_BIG.index[0], 
        accum_BIG.index[-1]])
    ax2.set_ylabel('# traces')

    fig.add_subplot(ax1)
    fig.add_subplot(ax2)

    if core_count.sum():
        ax3=ax2.twinx()
        core_count.plot(
            ax=ax3, color='grey', linewidth=2, 
            linestyle='--')
        ax3.set_ylabel('# Cores')
        fig.add_subplot(ax3)

fig.show()
# fig.savefig('Figuresbig-ts.pdf', bbox_inches='tight')
# %%

gdf_groups = gpd.GeoDataFrame(
    {'Group_ID': list(alpha_dict.values())[
        0:len(poly_groups)]}, 
    geometry=poly_groups, crs=gdf_BIG.crs)
# gdf_groups = gpd.GeoDataFrame(
#     {'Group_ID': alpha_dict.values()}, 
#     geometry=poly_groups, crs=gdf_BIG.crs)

bounds = {
    'x_range': tuple(gdf_BIG.total_bounds[0::2]), 
    'y_range': tuple(gdf_BIG.total_bounds[1::2])}

# Grid groups
group_plt = gv.Polygons(
    gdf_groups, vdims='Group_ID', 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        color_index='Group_ID', cmap=G_cmap, 
        tools=['hover'], fill_alpha=0.5)
grp_labs = hv.Labels(
    {('x', 'y'): np.array(
        [gdf_groups.geometry.centroid.x.values, 
        gdf_groups.geometry.centroid.y.values]).T, 
    'text': ['A', 'B', 'C', 'D']}, 
    ['x', 'y'], 'text').opts(
        text_font_size='22pt', text_color='red')

# Big grid cell plot
grid_plt = gv.Polygons(
    gdf_BIG, 
    crs=ANT_proj).opts(
        projection=ANT_proj, 
        line_color='black', color_index=None, 
        color=None)

# Radar data plot
radar_plt = gv.Polygons(
    gdf_grid_ALL, crs=ANT_proj).opts(
        projection=ANT_proj, 
        line_color=None, color='black')

# Core data plot
core_plt = gv.Points(
    gdf_cores, crs=ANT_proj, 
    vdims=['Name', 'size']).opts(
        projection=ANT_proj, color='blue', 
        line_color='black', marker='triangle', 
        size='size', tools=['hover'])

# Add plot to workspace
(
    group_plt * grid_plt * radar_plt 
    * grp_labs * core_plt).opts(
        width=700, height=700, 
        xlim=bounds['x_range'], 
        ylim=bounds['y_range'], bgcolor='silver')

# %%
