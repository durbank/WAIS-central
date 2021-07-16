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
from cartopy import feature as cf
# from bokeh.io import output_notebook
# output_notebook()
# hv.extension('bokeh')
# gv.extension('bokeh')
from shapely.geometry import Polygon, Point
import xarray as xr
from xrspatial import hillshade

# hv.archive.auto()

# Set project root directory
ROOT_DIR = Path(__file__).parents[1]

# Set project data directory
DATA_DIR = ROOT_DIR.joinpath('data/PAIPR-outputs')

# Import custom project functions
import sys
SRC_DIR = ROOT_DIR.joinpath('src')
sys.path.append(str(SRC_DIR))
from my_mods import paipr, stats
import my_mods.spat_ops as so

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(
    true_scale_latitude=-71)

# # Define Antarctic boundary file
# shp = str(ROOT_DIR.joinpath(
#     'data/Ant_basemap/Coastline_medium_res_polygon.shp'))

# Create Antarctic coastline polygon
world = gpd.read_file(
    gpd.datasets.get_path('naturalearth_lowres'))
Ant = world.query('continent=="Antarctica"')
Ant = Ant.to_crs(epsg=3031)

#%% Import and format PAIPR-generated results

# Import raw data
data_list = [folder for folder in DATA_DIR.glob('*')]
data_raw = pd.DataFrame()
for folder in data_list:
    data = paipr.import_PAIPR(folder)
    data_raw = data_raw.append(data)

# Remove results for below QC data reliability
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)

# Format and sort results for further processing
data_form = paipr.format_PAIPR(data_0)

# Create time series arrays for annual accumulation 
# and error
accum_ALL = data_form.pivot(
    index='Year', columns='trace_ID', values='accum')
std_ALL = data_form.pivot(
    index='Year', columns='trace_ID', values='std')

# Create gdf of mean results for each trace and 
# transform to Antarctic Polar Stereographic
gdf_traces = paipr.long2gdf(data_form)
gdf_traces.to_crs(epsg=3031, inplace=True)


# %% Get ice velocities of trace locations

# xr_vice = xr.open_dataset(ROOT_DIR.joinpath(
#     'data/ice-velocities/antarctica_ice_velocity_450m_v2.nc'))
# xr_pts = so.extract_at_pts(xr_vice, gdf_traces, return_dist=True)

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
gdf_cores = core_locs.loc[keep_idx,:]
core_ACCUM = core_ALL.loc[:,keep_idx].sort_index()

# Remove cores with less than 5 years of data
gdf_cores = gdf_cores.query('Duration >= 10')

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

# %% Study site inset plot

# gdf_bounds = {
#     'x_range': tuple(gdf_grid.total_bounds[0::2]+[-25000,25000]), 
#     'y_range': tuple(gdf_grid.total_bounds[1::2]+[-30000,25000])}
# gdf_bounds = {
#     'x_range': tuple(gdf_grid_ALL.total_bounds[0::2]), 
#     'y_range': tuple(gdf_grid_ALL.total_bounds[1::2])}
    
gdf_bounds = {
    'x_range': (-1.5E6-25E3, -9.9E5+25E3),
    'y_range': (-4.8E5-25E3, -5E4+25E3)}

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
Ant_bnds = gv.Polygons(
    Ant, crs=ANT_proj).opts(
        projection=ANT_proj, color='gray', 
        line_color=None)
# Ant_bnds = gv.Shape.from_shapefile(
#     shp, crs=ANT_proj).opts(
#         projection=ANT_proj, color='silver')

# Boundaries of core locations
core_bounds = {
    'x_range': bbox.bounds[0::2], 
    'y_range': bbox.bounds[1::2]}

# Lat/lon lines for inset
cf_big_grat = cf.NaturalEarthFeature(
    category='physical',
    name='graticules_15',
    scale='110m')
grats_big = gv.Feature(
    cf_big_grat, group='Lines').opts(
        gv.opts.Feature('Lines', projection=ANT_proj))

# Create inset map
inset_map = (
    Ant_bnds * radar_bnds 
    * grats_big.opts(line_color='white')).opts(
    width=700, height=700, bgcolor='lightsteelblue')

# %% Topography data

# Define Antarctic DEM file
xr_DEM = xr.open_rasterio(
    ROOT_DIR.joinpath(
        'data/Antarctica_Cryosat2_1km_DEMv1.0.tif')).squeeze()

# Clip elevation data to radar bounds
xr_DEM = xr_DEM.sel(
    x=slice(poly_bnds.bounds[0], poly_bnds.bounds[2]), 
    y=slice(poly_bnds.bounds[3],poly_bnds.bounds[1]))

# Generate elevation contours plot
elev_plt = hv.Image(xr_DEM.values, bounds=poly_bnds.bounds)
cont_plt = hv.operation.contours(elev_plt, levels=15).opts(
    cmap='kbc', show_legend=False, colorbar=True)

# Generate elevation hillshade
xr_HS = hillshade(xr_DEM)
hill_plt = hv.Image(
    xr_HS.values, bounds=poly_bnds.bounds).opts(
        alpha=0.25, cmap='dimgray', colorbar=False)

# %%

# Combine trace time series based on grid cells
grid_res = 2500
tmp_grids = so.pts2grid(gdf_traces, resolution=grid_res)
(gdf_grid_ALL, accum_grid_ALL, 
    MoE_grid_ALL, yr_count_ALL) = so.trace_combine(
    tmp_grids, accum_ALL, std_ALL)

# Limit time series to those within the bounds
poly_idx = gdf_grid_ALL.within(poly_bnds)
gdf_grid_ALL = gdf_grid_ALL.loc[poly_idx]
accum_grid_ALL = accum_grid_ALL[gdf_grid_ALL.index]
MoE_grid_ALL = MoE_grid_ALL[gdf_grid_ALL.index]
yr_count_ALL = yr_count_ALL[gdf_grid_ALL.index]

# Limit full time series to those with 10+ years of records
gdf_grid_ALL['Duration'] = accum_grid_ALL.notna().sum()
gdf_grid_ALL.query('Duration >= 10', inplace=True)
accum_grid_ALL = accum_grid_ALL[gdf_grid_ALL.index]
MoE_grid_ALL = MoE_grid_ALL[gdf_grid_ALL.index]
yr_count_ALL = yr_count_ALL[gdf_grid_ALL.index]

# Subset results to period 1979-2010
yr_start = 1979
yr_end = 2009
keep_idx = np.invert(
    accum_grid_ALL.loc[yr_start:yr_end,:].isnull().any())
accum_grid = accum_grid_ALL.copy().loc[
    yr_start:yr_end,keep_idx]
MoE_grid = MoE_grid_ALL.copy().loc[yr_start:yr_end,keep_idx]
gdf_grid = gdf_grid_ALL.copy().loc[keep_idx,:]
gdf_grid['accum'] = accum_grid.mean()
gdf_grid['MoE'] = MoE_grid.mean()

# Subset cores to same time period
keep_idx = np.invert(
    core_ACCUM.loc[yr_start:yr_end,:].isnull().any())
accum_core = core_ACCUM.loc[yr_start:yr_end,keep_idx]
gdf_core1979 = gdf_cores.copy().loc[keep_idx,:]
gdf_core1979['accum'] = accum_core.mean()

# %% Calculate linear trends

# Calculate trends in radar
trends, _, lb, ub = stats.trend_bs(
    accum_grid, 1000, df_err=MoE_grid)
gdf_grid['trend'] = trends
gdf_grid['t_lb'] = lb
gdf_grid['t_ub'] = ub
gdf_grid['t_perc'] = 100 * trends / gdf_grid['accum']

# Calculate trends in cores
trends, _, lb, ub = stats.trend_bs(accum_core, 1000)
gdf_core1979['trend'] = trends
gdf_core1979['t_lb'] = lb
gdf_core1979['t_ub'] = ub
gdf_core1979['t_perc'] = 100 * trends/gdf_core1979['accum']

# %% Calculate trends for partial-coverage cores

# Keep all core data from start of selected radar data to present
cores_long = core_ACCUM.loc[yr_start:]
gdf_long = gdf_cores.copy()
gdf_long['accum'] = cores_long.mean()

# Create size variable based on number of years in record
tmp = np.invert(
    cores_long.isna()).sum()
tmp = np.exp((tmp-tmp.min())/tmp.max())
gdf_long['size'] = 13*tmp

# Calculate trends in cores
trends, _, lb, ub = stats.trend_bs(cores_long, 1000)
gdf_long['trend'] = trends
gdf_long['t_lb'] = lb
gdf_long['t_ub'] = ub
gdf_long['t_perc'] = 100 * trends / gdf_long['accum']

# Determine trend significance for cores
gdf_long['sig'] = np.invert(np.array(
    [(gdf_long['t_lb'] < 0).values, 
    (gdf_long['t_ub'] > 0).values]).all(axis=0))

# Add additional trend column (for when I want to scale plot separately)
gdf_long['trend_plt'] = gdf_long['trend']

# %% Calculate trends for full duration

# trends, _, lb, ub = stats.trend_bs(
#     accum_grid_ALL, 1000, df_err=std_grid_ALL)
# gdf_grid_ALL['trend'] = trends
# gdf_grid_ALL['t_lb'] = lb
# gdf_grid_ALL['t_ub'] = ub
# gdf_grid_ALL['t_perc'] = 100*trends / gdf_grid_ALL['accum']

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

# %% Data location map

# Radar location plot
radar_plt = gv.Polygons(
    gdf_grid_ALL, crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        color='gray', alpha=0.75)

# Core location plot
core_plt = gv.Points(
    gdf_long, crs=ANT_proj, 
    vdims=['Name']).opts(
        projection=ANT_proj, color='blue', size=20, 
        line_color='black', marker='triangle', 
        tools=['hover'])

# Add to workspace
grat_data = cf.NaturalEarthFeature(
    category='physical',
    name='graticules_5',
    scale='110m')

graticules = gv.Feature(grat_data, group='Lines').opts(
        gv.opts.Feature('Lines', projection=ANT_proj))

data_map = (
    # elev_plt.opts(
    #     cmap='bmy', 
    #     # alpha=0.5,
    #     colorbar=True)
    # * hill_plt.opts(    
    hill_plt.opts(
        cmap='dimgray', 
        alpha=0.25, 
        colorbar=False)
    * cont_plt.opts(
        cmap='kbc',
        show_legend=False, 
        colorbar=True)
    * radar_plt.opts(color='red') 
    * core_plt
    * graticules.opts(line_color='white')
    ).opts(
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'], 
        width=700, height=700)

# %% Mean accum plots

# Plots for 1979-2010 period
accum_plt = gv.Polygons(
    gdf_grid, 
    vdims=['accum', 'MoE'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='viridis', colorbar=True, 
        tools=['hover'])
accum_core_plt = gv.Points(
    data=gdf_core1979, 
    vdims=['Name', 'accum'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, 
        line_color='black', size=20, 
        marker='triangle', tools=['hover'])
count_plt = gv.Polygons(
    gdf_grid, vdims='trace_count', 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='magma', colorbar=True, 
        tools=['hover'])

# Plot of mean accumulation across full time
aALL_plt = gv.Polygons(
    gdf_grid_ALL, vdims=['accum', 'MoE'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, 
        line_color=None, tools=['hover'])
aALL_core_plt = gv.Points(
    gdf_long, vdims=['Name', 'accum', 'size'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, 
        line_color='black', size='size', 
        marker='triangle', tools=['hover'])

# Plot of time series durations
duration_plt = gv.Polygons(
    gdf_grid_ALL, vdims=['Duration', 'trace_count'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='Duration', 
        cmap='magma', colorbar=True, 
        line_color=None, tools=['hover'])

# %%

# Calculate % change between mean accum 1979-2010 
# and mean accum for max duration available
ALL_subset = gdf_grid_ALL.loc[
    gdf_grid_ALL['grid_ID'].isin(gdf_grid['grid_ID'])]
res_accum = (ALL_subset['accum'] - gdf_grid['accum'])
rmse_accum = np.sqrt(
    (res_accum**2).sum() / (res_accum.count()-1))

res_gdf = gpd.GeoDataFrame(
    data={'res_accum':100*res_accum/gdf_grid.accum}, 
    geometry=ALL_subset.geometry, 
    crs=ALL_subset.crs)

res_tmp = gv.Polygons(
    data=res_gdf, vdims=['res_accum'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='res_accum', 
        cmap='PRGn', colorbar=True, symmetric=True,
        line_color=None, tools=['hover'])

# %% Final formating of accum plots

# All accum plot
# Calculate limits for colormap
ac_max = np.quantile(gdf_grid_ALL.accum, 0.99)
ac_min = np.quantile(gdf_grid_ALL.accum, 0.01)
AllAccum_plt = (
    hill_plt
    # * cont_plt.opts(colorbar=False)
    # elev_plt.opts(cmap='dimgray', colorbar=False)
    * (aALL_plt*aALL_core_plt).redim.range(
        accum=(ac_min,ac_max))
    ).opts(
        width=700, height=700,  
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])

# 1979-2010 accum plot
ac_max = gdf_grid.accum.max()
ac_min = gdf_grid.accum.min()
accum1979_plt = (
    hill_plt
    # elev_plt.opts(cmap='dimgray', colorbar=False)
    # * cont_plt.opts(colorbar=False)
    * radar_plt.opts(color='black', alpha=0.5)
    * (accum_plt*accum_core_plt).redim.range(
        accum=(250,500))
    ).opts(
        width=700, height=700, 
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])

# Record duration plot
ALLduration_plt = (
    hill_plt
    # * cont_plt.opts(colorbar=False)
    * duration_plt).opts(
        width=700, height=700, 
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])

# Residual between accum plots
res_min = np.quantile(res_gdf.res_accum, 0.01)
res_max = np.quantile(res_gdf.res_accum, 0.99)
res_plt = (
    # hill_plt
    # * cont_plt.opts(colorbar=False)
    radar_plt
    * res_tmp.redim.range(res_accum=(res_min,res_max))
    ).opts(
        bgcolor='silver',
        width=700, height=700,  
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])

# %% Deprecated code for MoE of trend results

gdf_ERR = gdf_grid.copy().drop(
    ['accum', 'MoE', 't_lb', 
    't_ub', 't_perc'], axis=1)

gdf_ERR['t_MoE'] = pd.DataFrame(
    [gdf_grid['t_ub']-gdf_grid['trend'], 
    gdf_grid['trend']-gdf_grid['t_lb']]).max()
# gdf_ERR['t_MoE'] = (
#     gdf_grid['t_ub'] 
#     - gdf_grid['t_lb']) / 2

tERR_plt = gv.Polygons(
    gdf_ERR, vdims=['t_MoE', 'trend'], 
    crs=ANT_proj).opts(projection=ANT_proj, 
    line_color=None, cmap='plasma', colorbar=True, 
    tools=['hover'])

tMOE_plt = (
    # hill_plt
    cont_plt.opts(colorbar=False, bgcolor='silver')
    * tERR_plt).opts(
        width=700, height=700, 
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])

# %% Trend plots only including cores with complete 1979-2010 temporal coverage

# Add boolean for trend signficance
insig_idx = gdf_grid.query(
    't_lb<0 & t_ub>0').index.values
sig_idx = np.invert(np.array(
    [(gdf_grid['t_lb'] < 0).values, 
    (gdf_grid['t_ub'] > 0).values]).all(axis=0))

# Add boolean for trend significance
gdf_core1979['sig'] = np.invert(np.array(
    [(gdf_core1979['t_lb'] < 0).values, 
    (gdf_core1979['t_ub'] > 0).values]).all(axis=0))

# Radar trend plot
t_plt = gv.Polygons(
    data=gdf_grid, 
    vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, tools=['hover'])

# Core trend plot
core_t_plt = gv.Points(
    data=gdf_core1979[gdf_core1979.within(poly_bnds)], 
    vdims=['Name', 'trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, size=20, line_color='black', 
        marker='triangle', tools=['hover'])

# Sig/insig radar plots
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

# Sig/insig core plots
sig_core_plt = gv.Points(
    gdf_core1979[gdf_core1979.within(poly_bnds)].query(
        'sig==True'), 
    vdims=['Name', 'trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, size=20, line_color='black', 
        marker='triangle', tools=['hover'])
insig_core_plt = gv.Points(
    gdf_core1979[gdf_core1979.within(poly_bnds)].query(
        'sig==False'), 
    vdims=['Name', 'trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='grey', alpha=0.75,  
        size=20, line_color='black', 
        marker='triangle', tools=['hover'])

# Radar trend (as percent) plot
tPERC_plt = gv.Polygons(
    data=gdf_grid, 
    vdims=['t_perc'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, tools=['hover'])

# Core trend (as percent) plot
core_tPERC_plt = gv.Points(
    data=gdf_core1979[gdf_core1979.within(poly_bnds)], 
    vdims=['Name', 't_perc'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='t_perc', 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, size=20, line_color='black', 
        marker='triangle', tools=['hover'])

# Core trends (as percent) for wider WAIS region
coreTMP_plt = gv.Points(
    data=gdf_long, 
    vdims=[
        'Name', 'trend_plt', 't_perc',
        't_lb', 't_ub', 'size'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='t_perc', 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, size='size', 
        line_color='black', 
        marker='triangle', tools=['hover'])

# %% Final formatting for above plots

t_max = np.max(
    [gdf_grid.trend.max(), 
    gdf_core1979[gdf_core1979.within(
        poly_bnds)].trend.max()])
t_min = np.min(
    [gdf_grid.trend.min(), 
    gdf_core1979[gdf_core1979.within(
        poly_bnds)].trend.min()])

# All trends plot
trend_plt = (
    # hill_plt
    # elev_plt.opts(cmap='dimgray', colorbar=False)
    # cont_plt.opts(colorbar=False)*
    t_plt
    * core_t_plt.redim.range(trend=(-15,15))).opts(
        bgcolor='silver',
        width=700, height=700, 
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])

# Sig trends plot
trend_sig_plt = (
    # hill_plt.opts(cmap='dimgray', alpha=0.33, colorbar=False)
    # cont_plt.opts(colorbar=False)*
    insig_plt * sig_plt
    * insig_core_plt * sig_core_plt).opts(
        bgcolor='silver',
        width=700, height=700, 
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])

# Percent change trend plot
# tPERC_max = np.quantile(gdf_grid['t_perc'], 0.99)
# tPERC_min = np.quantile(gdf_grid['t_perc'], 0.01)
trendPERC_plt = (
    # hill_plt.opts(cmap='dimgray', alpha=0.33, colorbar=False)
    # elev_plt.opts(cmap='dimgray', colorbar=False)
    # cont_plt.opts(colorbar=False)*
    tPERC_plt
    * core_tPERC_plt).opts(
        bgcolor='silver', 
        width=700, height=700, 
        xlim=gdf_bounds['x_range'], 
        ylim=gdf_bounds['y_range'])

# Percent change trends plot for WAIS-wide cores
coreWIDE_plt = (
    Ant_bnds
    *radar_plt.opts(color='black', alpha=1)
    * coreTMP_plt.opts(
        bgcolor='lightsteelblue')
    ).opts(
        xlim=core_bounds['x_range'], 
        ylim=core_bounds['y_range'], 
        width=700, height=700)



# %% Larger grid cells and composite time series

# Keep all core data from start of selected radar data to present
cores_accum1960 = core_ACCUM.loc[1960::]
gdf_cores['accum'] = cores_accum1960.mean()

# Create size variable based on number of years in record
tmp = np.invert(
    cores_accum1960.isna()).sum()
tmp = np.exp((tmp-tmp.min())/tmp.max())
gdf_cores['size'] = 13*tmp

# Combine trace time series based on grid cells
tmp_grid_big = so.pts2grid(gdf_traces, resolution=100000)
(gdf_BIG, accum_BIG, 
    MoE_BIG, yr_count_BIG) = so.trace_combine(
    tmp_grid_big, accum_ALL, std_ALL)

# Add the total duration of the time series for each grid
gdf_BIG['Duration'] = accum_BIG.notna().sum()

# Limit time series to those within the bounds
poly_idx = gdf_BIG.intersects(poly_bnds)
gdf_BIG = gdf_BIG.loc[poly_idx]

# Limit to those with a duration greater than 30 years
gdf_BIG = gdf_BIG.query('Duration > 30')

# Remove grid cells with fewer than 100 raw traces
gdf_BIG = gdf_BIG.query('trace_count >= 100')

# Filter removed data from other variables
accum_BIG = accum_BIG[gdf_BIG.index]
MoE_BIG = MoE_BIG[gdf_BIG.index]
yr_count_BIG = yr_count_BIG[gdf_BIG.index]

# Calculate trends for large grid cells
trends, _, lb, ub = stats.trend_bs(
    accum_BIG, 1000, df_err=MoE_BIG)
gdf_BIG['trend'] = trends
gdf_BIG['t_lb'] = lb
gdf_BIG['t_ub'] = ub
gdf_BIG['t_perc'] = 100*trends / gdf_BIG['accum']

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

BC_plt = gv.Polygons(
    data=gdf_BIG, 
    vdims=['trace_count', 'Duration'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None,
        cmap='magma', colorbar=True, 
        tools=['hover'])

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

# Calculate colormap limits for trend margin of error
ERR_BIG = gdf_BIG.copy().drop(
    ['accum', 'MoE', 't_lb', 
    't_ub', 't_perc'], axis=1)
ERR_BIG['t_MoE'] = pd.DataFrame(
    [gdf_BIG['t_ub']-gdf_BIG['trend'], 
    gdf_BIG['trend']-gdf_BIG['t_lb']]).max()
# ERR_BIG['t_MoE'] = (
#     gdf_BIG['t_ub'] 
#     - gdf_BIG['t_lb']) / 2
ERR_max = np.quantile(ERR_BIG.t_MoE, 0.99)
ERR_min = np.quantile(ERR_BIG.t_MoE, 0.01)

# Margin of error on trends plot
ERR_BIG_plt = gv.Polygons(
    ERR_BIG, vdims=['t_MoE', 'trend'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='plasma', colorbar=True, tools=['hover'])

# %% Final formatting of plots

trendBig_plt = (
    # cont_plt.opts(colorbar=False)*
    tBIG_plt*sig_plt
    *radar_plt.opts(color='gray', alpha=0.75)).opts(
        width=700, height=700
        , bgcolor='silver'
        ).redim.range(
        trend=(tr_min,tr_max))

moeBig_plt = (
    # elev_plt.opts(cmap='dimgray', colorbar=False)*
    ERR_BIG_plt.redim.range(MoE=(ERR_min,ERR_max))
    * radar_plt).opts(
        width=700, height=700
        , bgcolor='silver'
    )

BigCount_plt = (
    BC_plt
    *radar_plt.opts(color='white', alpha=1)).opts(
    width=700, height=700, bgcolor='silver')

# %%

grp_A = [22, 23, 29, 35]
grp_B = [27, 28, 33, 34, 39, 40]
grp_C = [19, 20, 21, 25, 26, 31, 32]
grp_D = [37, 38, 43, 49]

gdf_BIG['Group'] = ""

for i, group in enumerate(
    [grp_A, grp_B, grp_C, grp_D]):

    row_idx = gdf_BIG['grid_ID'].isin(group)
    gdf_BIG.loc[row_idx,'Group'] = i

alpha_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
gdf_BIG.replace({'Group': alpha_dict}, inplace=True)

# %%

poly_groups = []
G_cmap = {
    'A':'#66C2A5', 'B':'#FC8D62', 
    'C':'#8DA0CB', 'D':'#E78AC3'}

plt.rcParams.update({'font.size': 18})
BigTS_fig = plt.figure(figsize=(13, 10))
outer = gridspec.GridSpec(
    2, 2, wspace=0.5, hspace=0.4)

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
    ax1 = plt.Subplot(BigTS_fig, inner[0])
    ax2 = plt.Subplot(BigTS_fig, inner[1])
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
    ax2.set_ylabel('No. traces')

    BigTS_fig.add_subplot(ax1)
    BigTS_fig.add_subplot(ax2)

    if core_count.sum():
        ax3=ax2.twinx()
        core_count.plot(
            ax=ax3, color='grey', linewidth=2, 
            linestyle='--')
        ax3.set_ylabel('No. Cores')
        BigTS_fig.add_subplot(ax3)

# BigTS_fig.show()
# fig.savefig('Figuresbig-ts.pdf', bbox_inches='tight')
# %%

gdf_groups = gpd.GeoDataFrame(
    {'Group_ID': list(alpha_dict.values())[
        0:len(poly_groups)]}, 
    geometry=poly_groups, crs=gdf_BIG.crs)

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
        text_font_size='48pt', 
        text_font_style='bold',
        text_color='white')

# Big grid cell plot
grid_plt = gv.Polygons(
    gdf_BIG, 
    crs=ANT_proj).opts(
        projection=ANT_proj, 
        line_color='black', color_index=None, 
        color=None)

# Core data plot
coreBig_plt = gv.Points(
    gdf_cores, crs=ANT_proj, 
    vdims=['Name', 'size']).opts(
        projection=ANT_proj, color='blue', 
        line_color='black', marker='triangle', 
        size='size', tools=['hover'])

# Add plot to workspace
group_map = (
    group_plt * grid_plt 
    * radar_plt
    * coreBig_plt * grp_labs).opts(
        width=700, height=700, 
        xlim=bounds['x_range'], 
        ylim=bounds['y_range'], 
        bgcolor='silver')

# %% Output calculated values of interest for manuscript

# Only output values if using the appropriate resolution
if grid_res == 1000:

    # Bias and RMSE between mean accum values
    print(f"Mean bias between 1979-2010 mean accum and mean accum for full available duration is {res_accum.mean():.2f} mm/yr ({100*(res_accum/gdf_grid['accum']).mean():.2f}%)")
    print(f"RMSE between 1979-2010 mean accum and mean for full duration is {rmse_accum:.2f} mm/yr ({100*rmse_accum/gdf_grid['accum'].mean():.2f}%)")

    # Number of data points in data set
    print(f"Total individual estimates of annual accumulation: {accum_ALL.count().sum()}")
    print(f"Total number of 1-km time series with 10+ years of coverage: {gdf_grid_ALL.shape[0]}")
    print(f"Number of time series with full coverage 1979-2010: {gdf_grid.shape[0]}")

    # Result significance fractions
    print(f"Results with significant negative trends: {100*gdf_grid[sig_idx].query('trend<0').shape[0]/gdf_grid.shape[0]:.1f}%")
    print(f"Results with significant positive trends: {100*gdf_grid[sig_idx].query('trend>0').shape[0]/gdf_grid.shape[0]:.1f}%")

    # 95% bounds for trend results
    tBND_up = 10*np.quantile(gdf_grid['trend'], 0.975)
    tBND_low = 10*np.quantile(gdf_grid['trend'], 0.025)
    tUP_perc = 10*np.quantile(gdf_grid['t_perc'], 0.975)
    tLOW_perc = 10*np.quantile(gdf_grid['t_perc'], 0.025)
    print(f"95% of accumulation trend magnitudes fall within the range {tBND_low:.2f} to {tBND_up:.2f} mm/decade ({tLOW_perc:.2f}% to {tUP_perc:.2f}% per decade)")

    # Area-integrated trend in SMB for full data set (1979-2010)
    AIT_mu = gdf_grid['trend'].mean()
    AIT_MoE = 1.96*np.sqrt(gdf_grid['trend'].std()/gdf_grid.shape[0])
    print(f"Area-integrated trend for full data set: {AIT_mu:.2f}+/-{AIT_MoE:.2f} mm/a ({gdf_grid['t_perc'].mean():.2f}%+/-{1.96*np.sqrt(gdf_grid['t_perc'].std()/gdf_grid.shape[0]):.2f}%)")

    # Significance fractions for gdf_BIG
    print(f"Large grid cells with significant negative trends: {100*gdf_BIG[sig_BIG].query('trend<0').shape[0]/gdf_BIG.shape[0]:.2f}%")
    print(f"Large grid cells with significant positive trends: {100*gdf_BIG[sig_BIG].query('trend>0').shape[0]/gdf_BIG.shape[0]:.2f}%")

# %%

# elev_plt = elev_plt.opts(colorbar=True)
cont_plt.opts(colorbar=True)
radar_plt.opts(color='red', alpha=1)
data_map = data_map.opts(fontscale=2)
data_panel = (inset_map + data_map)

# %%

if grid_res > 1000:
    hv.save(inset_map, ROOT_DIR.joinpath(
        'docs/Figures/inset.png'))
    hv.save(data_map, ROOT_DIR.joinpath(
        'docs/Figures/data_map.png'))

# %%

# elev_plt = elev_plt.opts(colorbar=False)
cont_plt.opts(colorbar=False)
radar_plt.opts(color='gray', alpha=0.75)
accum_panel = hv.Layout(
    AllAccum_plt.opts(
        height=1000, width=1000, fontscale=3)
    + ALLduration_plt.opts(
        height=1000, width=1000, fontscale=3)
    + accum1979_plt.opts(
        width=1000, height=1000, fontscale=3)
    + res_plt.opts(
        width=1000, height=1000, fontscale=3)).cols(2)

trend_panel = hv.Layout(
    trend_plt.opts(
        height=1000, width=1000, fontscale=3)
    + trendPERC_plt.opts(
        height=1000, width=1000, fontscale=3)
    + trend_sig_plt.opts(
        height=1000, width=1000, fontscale=3)
    ).cols(2).redim.range(trend=(-8,8), t_perc=(-3,3))

BIG_panel = hv.Layout(
    group_map.opts(
        height=1000, width=1000, fontscale=3)
    + trendBig_plt.opts(
        height=1000, width=1000, fontscale=3)
    + moeBig_plt.opts(
        height=1000, width=1000, fontscale=3)
    + BigCount_plt.opts(
        height=1000, width=1000, fontscale=3)).cols(2)

# %%

if grid_res > 1000:
    hv.save(accum_panel, ROOT_DIR.joinpath(
        'docs/Figures/accum_maps.png'))
    hv.save(trend_panel, ROOT_DIR.joinpath(
        'docs/Figures/trend_maps.png'))
    hv.save(BIG_panel, ROOT_DIR.joinpath(
        'docs/Figures/BIG_maps.png'))

# %%

radar_plt.opts(color='black', alpha=1)
trendPERC_panel = coreWIDE_plt.opts(
        height=1000, width=1000, fontscale=3).redim.range(
            t_perc=(-3,3))

# %%

if grid_res > 1000:
    hv.save(trendPERC_panel, ROOT_DIR.joinpath(
        'docs/Figures/trendPERC_maps.png'))    
    BigTS_fig.savefig(fname=ROOT_DIR.joinpath(
        'docs/Figures/BigTS_fig.svg'))
