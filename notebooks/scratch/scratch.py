# This script currently is for investigating changes to PAIPR results using gamma, Gaussian, and mixture model distributions

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
import matplotlib.pyplot as plt

# Set project root directory
ROOT_DIR = Path(__file__).parents[2]

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

# %% Get gamma dataframe

# # Import gamma results
# data_dir = ROOT_DIR.joinpath('data/gamma/20111109/')
# data_raw = import_PAIPR(data_dir)

# # Subset data into QC flags 0, 1, and 2
# data_0 = data_raw[data_raw['QC_flag'] == 0]
# data_1 = data_raw[data_raw['QC_flag'] == 1]
# # data_2 = data_raw[data_raw['QC_flag'] == 2]

# # Remove data_1 values earlier than assigned QC yr, 
# # and recombine results with main data results
# data_1 = data_1[data_1.Year >= data_1.QC_yr]
# data_0 = data_0.append(data_1).sort_values(
#     ['collect_time', 'Year'])

# # Format accumulation data
# accum_long = format_PAIPR(
#     data_0, start_yr=1979, end_yr=2010).drop(
#         'elev', axis=1)

# # New accum and std dfs in wide format
# acc_gamma = accum_long.pivot(
#     index='Year', columns='trace_ID', values='accum')
# err_gamma = accum_long.pivot(
#     index='Year', columns='trace_ID', values='std')

# # Create df for mean annual accumulation
# accum_long['collect_time'] = (
#     accum_long.collect_time.values.astype(np.int64))
# traces_gamma = accum_long.groupby('trace_ID').mean().drop(
#     'Year', axis=1)
# traces_gamma['collect_time'] = (
#     pd.to_datetime(traces_gamma.collect_time)
#     .dt.round('1ms'))
# traces_gamma = gpd.GeoDataFrame(
#     traces_gamma, geometry=gpd.points_from_xy(
#         traces_gamma.Lon, traces_gamma.Lat), 
#     crs="EPSG:4326").drop(['Lat', 'Lon'], axis=1)

# # Convert accum crs to same as Antarctic outline
# traces_gamma = traces_gamma.to_crs(ant_outline.crs)

# # # Drop original index values
# # traces_gamma = traces_gamma.drop('index', axis=1)

# %% Get Gaussian dataframe

# Import Gaussian results
data_dir = ROOT_DIR.joinpath('data/Gauss/20111109/')
data_raw = import_PAIPR(data_dir)

# Subset data into QC flags 0, 1, and 2
data_0 = data_raw[data_raw['QC_flag'] == 0]
data_1 = data_raw[data_raw['QC_flag'] == 1]

# Remove data_1 values earlier than assigned QC yr, 
# and recombine results with main data results
data_1 = data_1[data_1.Year >= data_1.QC_yr]
data_0 = data_0.append(data_1).sort_values(
    ['collect_time', 'Year'])

# Format accumulation data
accum_long = format_PAIPR(
    data_0, start_yr=1979, end_yr=2010).drop(
        'elev', axis=1)

# New accum and std dfs in wide format
acc_gauss = accum_long.pivot(
    index='Year', columns='trace_ID', values='accum')
err_gauss = accum_long.pivot(
    index='Year', columns='trace_ID', values='std')

# Convert to gdf
traces_gauss = long2gdf(accum_long)

# Convert accum crs to same as Antarctic outline
traces_gauss = traces_gauss.to_crs(ant_outline.crs)

# %% Get Gaussian mixture dataframe

# Import Gaussian mixture results
data_dir = Path(
    '/media/durbank/WARP/Research/Antarctica/Data/'
    + 'CHPC/PAIPR-results/2020-10-07/Outputs/20111109/')
data_raw = import_PAIPR(data_dir)

# Subset data into QC flags 0, 1, and 2
data_0 = data_raw[data_raw['QC_flag'] == 0]
data_1 = data_raw[data_raw['QC_flag'] == 1]

# Remove data_1 values earlier than assigned QC yr, 
# and recombine results with main data results
data_1 = data_1[data_1.Year >= data_1.QC_yr]
data_0 = data_0.append(data_1).sort_values(
    ['collect_time', 'Year'])

# Format accumulation data
accum_long = format_PAIPR(
    data_0, start_yr=1979, end_yr=2010).drop(
        'elev', axis=1)

# New accum and std dfs in wide format
acc_mix = accum_long.pivot(
    index='Year', columns='trace_ID', values='accum')
err_mix = accum_long.pivot(
    index='Year', columns='trace_ID', values='std')

# Convert to gdf
traces_mix = long2gdf(accum_long)

# Convert accum crs to same as Antarctic outline
traces_mix = traces_mix.to_crs(ant_outline.crs)

# %% Initial mean accum maps

# gamma_map = gv.Points(
#     traces_gamma.sample(5000), 
#     vdims=gv.Dimension('accum', range=(150,550)), 
#     crs=ANT_proj).opts(projection=ANT_proj, color='accum', 
#     cmap='viridis', colorbar=True, 
#     tools=['hover'], width=600, height=400)
gauss_map = gv.Points(
    traces_gauss, 
    vdims=gv.Dimension('accum', range=(150,550)), 
    crs=ANT_proj).opts(projection=ANT_proj, color='accum', 
    cmap='viridis', colorbar=True, 
    tools=['hover'], width=600, height=400)
mixture_map = gv.Points(
    traces_mix, 
    vdims=gv.Dimension('accum', range=(150,550)), 
    crs=ANT_proj).opts(projection=ANT_proj, color='accum', 
    cmap='viridis', colorbar=True, 
    tools=['hover'], width=600, height=400)
gauss_map + mixture_map

# %% Match all results to nearest corresponding results
# (Based on comparisons to Gauss mixture results)

df_dist = nearest_neighbor(
    traces_mix, traces_gauss, return_dist=True)
idx_mix = df_dist['distance'] <= 250
dist_gauss = df_dist[idx_mix]

# Create numpy arrays for relevant results
accum_gauss = acc_gauss.iloc[:,dist_gauss.index]
std_gauss = err_gauss.iloc[:,dist_gauss.index]
accum_mix = acc_mix[acc_mix.columns[idx_mix]]
std_mix = err_mix[err_mix.columns[idx_mix]]

# Add results to new combined dataframe
gdf_traces = gpd.GeoDataFrame(
    {'ID_gauss': accum_gauss.columns, 
    'ID_mix': accum_mix.columns, 
    'accum_gauss': accum_gauss.mean().values, 
    'accum_mix': accum_mix.mean().values},
    geometry=dist_gauss.geometry.values)

# %% Annual estimate residuals

res_yr = pd.DataFrame(
    accum_mix.to_numpy() - accum_gauss.to_numpy(), 
    index=accum_gauss.index)

# Create dataframes for annual scatter plots
accum_df = pd.DataFrame(
    {'Year': np.reshape(np.repeat(
        accum_gauss.index, accum_gauss.shape[1]), 
        accum_gauss.size), 
    'accum_gauss': \
        np.reshape(accum_gauss.to_numpy(), accum_gauss.size), 
    'std_gauss': np.reshape(std_gauss.to_numpy(), std_gauss.size), 
    'accum_mix': \
        np.reshape(accum_mix.to_numpy(), accum_mix.size), 
    'std_mix': np.reshape(std_mix.to_numpy(), std_mix.size)})

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=accum_df, 
    kdims=['accum_gauss', 'accum_mix'], 
    vdims=['Year']).groupby('Year')
(one_to_one.opts(color='black') 
    * scatt_yr.opts(xlim=(100,750), ylim=(100,750)))

# %%
# Calculate robust linear trends
t_gauss, intrcpt, lb_gauss, ub_gauss = trend_bs(
    accum_gauss, 1000)
t_mix, intrcpt, lb_mix, ub_mix = trend_bs(
    accum_mix, 1000)

# Add trend results to traces gdf
gdf_traces['trend_gauss'] = t_gauss.values
gdf_traces['lb_gauss'] = lb_gauss
gdf_traces['ub_gauss'] = ub_gauss
gdf_traces['trend_mix'] = t_mix.values
gdf_traces['lb_mix'] = lb_mix
gdf_traces['ub_mix'] = ub_mix


gdf_traces['trend_res'] = (gdf_traces['trend_mix'] 
    - gdf_traces['trend_gauss']) / (
    gdf_traces[['accum_gauss', 'accum_mix']].mean(axis=1))
gdf_traces['trend_res'].plot(kind='density')

# %% Mean accum 1:1 plot
one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[200,500], 'y':[200,500]}))
scatt_yr = hv.Points(
    data=pd.DataFrame(gdf_traces).drop(columns=['geometry']), 
    kdims=['accum_gauss', 'accum_mix'])
(one_to_one.opts(color='black') 
    * scatt_yr.opts(xlim=(200,500), ylim=(200,500)))

# %% Accum trend 1:1 plot
one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[-0.05,0.01], 'y':[-0.05,0.01]}))
scatt_yr = hv.Points(
    data=pd.DataFrame({
        'trend_gauss': 
        gdf_traces['trend_gauss']/gdf_traces['accum_gauss'], 
        'trend_mix': 
        gdf_traces['trend_mix']/gdf_traces['accum_mix']}), 
    kdims=['trend_gauss', 'trend_mix'])
(one_to_one.opts(color='black') 
    * scatt_yr.opts(xlim=(-0.05,0.01), ylim=(-0.05,0.01)))

# %%

Tgauss_plt = gv.Points(
    gdf_traces, 
    vdims=['trend_gauss', 'lb_gauss', 'ub_gauss'], 
    crs=ANT_proj). opts(
        projection=ANT_proj, color='trend_gauss', 
        cmap='coolwarm', symmetric=True, colorbar=True, 
        tools=['hover'], width=600, height=400)
Tmix_plt = gv.Points(
    gdf_traces, 
    vdims=['trend_mix', 'lb_mix', 'ub_mix'], 
    crs=ANT_proj). opts(
        projection=ANT_proj, color='trend_mix', 
        cmap='coolwarm', symmetric=True, colorbar=True, 
        tools=['hover'], width=600, height=400)
Tgauss_plt + Tmix_plt

# %%

gdf_traces['err_gauss'] = (
    np.mean([
        gdf_traces['ub_gauss'] - gdf_traces['trend_gauss'], 
        gdf_traces['trend_gauss'] - gdf_traces['lb_gauss']], 
        axis=0) 
    / gdf_traces['accum_gauss'])

Tres_plt = gv.Points(
    gdf_traces, 
    vdims=['trend_res'], 
    crs=ANT_proj). opts(
        projection=ANT_proj, color='trend_res', 
        cmap='coolwarm', symmetric=True, colorbar=True, 
        tools=['hover'], width=600, height=400)
Tres_plt

# %% Individual time series comparisons

import random as rnd

ii = rnd.sample(range(accum_gauss.shape[1]), 4)
for i in ii:
    r = rnd.random()
    b = rnd.random()
    g = rnd.random()
    color = (r, g, b)

    plt.plot(accum_gauss.iloc[:,i], color=color)
    plt.plot(accum_mix.iloc[:,i], color=color, linestyle='--')
plt.show()

# %% TS comparisons with errors

# ii = rnd.sample(range(accum_gauss.shape[1]), 10)
ii = np.arange(0,accum_gauss.shape[1], 250)
for i in ii:
    print(i)
    plt.figure()
    plt.plot(accum_gauss.iloc[:,i], color='blue')
    plt.plot(
        accum_gauss.iloc[:,i]+std_gauss.iloc[:,i], 
        color='blue', linestyle='--')
    plt.plot(
        accum_gauss.iloc[:,i]-std_gauss.iloc[:,i], 
        color='blue', linestyle='--')
    plt.plot(accum_mix.iloc[:,i], color='red')
    plt.plot(
        accum_mix.iloc[:,i]+std_mix.iloc[:,i], 
        color='red', linestyle='--')
    plt.plot(
        accum_mix.iloc[:,i]-std_mix.iloc[:,i], 
        color='red', linestyle='--')
    plt.show()

# %%

# # %%
# ## ACF exploration

# # 
# accum_acf = acf(accum)

# accum_acf.mean(axis=1).plot(color='blue', linewidth=2)
# (accum_acf.mean(axis=1) 
#     + accum_acf.std(axis=1)).plot(
#         color='blue', linestyle='--')
# (accum_acf.mean(axis=1) 
#     - accum_acf.std(axis=1)).plot(
#         color='blue', linestyle='--')

# # %%
# ## Plot data inset map
# Ant_bnds = gv.Shape.from_shapefile(
#     ant_path, crs=ANT_proj).opts(
#     projection=ANT_proj, width=500, height=500)
# trace_plt = gv.Points(accum_trace, crs=ANT_proj).opts(
#     projection=ANT_proj, color='red')
# Ant_bnds * trace_plt
# # %%

# accum_plt = gv.Points(
#     accum_trace, 
#     vdims=['accum', 'std'], 
#     crs=ANT_proj).opts(projection=ANT_proj, color='accum', 
#     cmap='viridis', colorbar=True, 
#     tools=['hover'], width=600, height=400)
# accum_plt

# # %%

# trend_plt = gv.Points(
#     accum_trace, 
#     vdims=['trend_perc', 't_lb', 't_ub'], 
#     crs=ANT_proj). opts(
#         projection=ANT_proj, color='trend_perc', 
#         cmap='coolwarm', symmetric=True, colorbar=True, 
#         tools=['hover'], width=600, height=400)
# trend_plt

# # %%
# Tabs_plt = gv.Points(
#     accum_trace, 
#     vdims=['trend_abs'], 
#     crs=ANT_proj). opts(
#         projection=ANT_proj, color='trend_abs', 
#         cmap='coolwarm', symmetric=False, colorbar=True, 
#         tools=['hover'], width=600, height=400)
# Tabs_plt

# # %%
# ## Plot random accumulation time series
# import matplotlib.pyplot as plt

# i = np.random.randint(accum.shape[1])
# yr = accum.index
# smb = accum.iloc[:,i]
# smb_err = accum_std.iloc[:,i]
# fig, ax = plt.subplots()
# ax.plot(yr, smb, color='red', lw=2)
# ax.plot(yr, smb+smb_err, color='red', ls='--')
# ax.plot(yr, smb-smb_err, color='red', ls='--')
# fig.show()

# %%
