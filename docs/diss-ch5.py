# Code used to generate analyses and plots to be included in Ch.5 of doctoral dissertation
# This principally focuses on a spectral and time series analyses of perodicities in SMB variability in WAIS

# %%

# Import requisite modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import geoviews as gv
import holoviews as hv
from cartopy import crs as ccrs
from shapely.geometry import Point, Polygon
hv.extension('bokeh', 'matplotlib')
gv.extension('bokeh', 'matplotlib')
from scipy import signal

# Set project root directory
ROOT_DIR = Path('/home/durbank/Documents/Research/Antarctica/WAIS-central/')

# Set project data directory
DATA_DIR = ROOT_DIR.joinpath('data')

# Import custom project functions
import sys
SRC_DIR = ROOT_DIR.joinpath('src')
sys.path.append(str(SRC_DIR))
from my_mods import paipr, stats
from my_mods import spat_ops as so

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)

# Import Antarctic outline shapefile
ant_path = DATA_DIR.joinpath(
    'Ant_basemap/Coastline_medium_res_polygon.shp')
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

# Flip to get results in tidy-compliant form
core_meta = core_meta.transpose()
core_meta.index.name = 'Name'
# core_ALL = core_ALL.transpose()

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
# gdf_cores = core_locs.copy()
# core_ACCUM = core_ALL.sort_index()

# # Remove cores with less than 5 years of data
# gdf_cores = gdf_cores.query('Duration >= 10')

# # Remove cores with missing elev data
# # (this only gets rid of Ronne ice shelf cores)
# gdf_cores = gdf_cores[gdf_cores['Elev'].notna()]

# # Remove specific unwanted cores
# gdf_cores.drop('SEAT-10-4', inplace=True)
# gdf_cores.drop('BER11C95_25', inplace=True)
# gdf_cores.drop('SEAT-11-1', inplace=True)
# gdf_cores.drop('SEAT-11-2', inplace=True)
# gdf_cores.drop('SEAT-11-3', inplace=True)
# gdf_cores.drop('SEAT-11-4', inplace=True)
# gdf_cores.drop('SEAT-11-6', inplace=True)
# gdf_cores.drop('SEAT-11-7', inplace=True)
# gdf_cores.drop('SEAT-11-8', inplace=True)

# Remove additional cores from core time series
core_ACCUM = core_ACCUM[gdf_cores.index]

#%% Import and format PAIPR-generated results

# Import raw data
data_list = [
    folder for folder in 
    DATA_DIR.joinpath('PAIPR-outputs').glob('*')]
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

# %% PAIPR data aggregation by grid

# Combine trace time series based on grid cells
grid_res = 1000
tmp_grids = so.pts2grid(gdf_traces, resolution=grid_res)
(gdf_grid_ALL, accum_grid_ALL, 
    MoE_grid_ALL, yr_count_ALL) = so.trace_combine(
    tmp_grids, accum_ALL, std_ALL)

# %% Subset results to standard time period

# Set start and end years
yr_start = 1979
yr_end = 2010

# Subset PAIPR data
keep_idx = np.invert(
    accum_grid_ALL.loc[yr_start:yr_end,:].isnull().any())
accum_grid = accum_grid_ALL.loc[yr_start:yr_end,keep_idx]
MoE_grid = MoE_grid_ALL.loc[yr_start:yr_end,keep_idx]
gdf_grid = gdf_grid_ALL.copy().loc[keep_idx,:]
gdf_grid['accum'] = accum_grid.mean()
gdf_grid['MoE'] = MoE_grid.mean()

# Subset cores to same time period
keep_idx = np.invert(
    core_ACCUM.loc[yr_start:yr_end,:].isnull().any())
accum_core = core_ACCUM.loc[yr_start:yr_end,keep_idx]
gdf_core = gdf_cores.copy().loc[keep_idx,:]
gdf_core['accum'] = accum_core.mean()

# %% Autocorrelation analysis

# core_acf = acf(accum_core)
# core_acf.plot()

# %%
## Orientation plot showing location of data

radar_plt = gv.Points(
    gdf_grid, crs=ANT_proj).opts(
        projection=ANT_proj, color='red')
core_plt = gv.Points(
    gdf_core, crs=ANT_proj, 
    vdims=['Name', 'Duration']).opts(
        projection=ANT_proj, color='blue', size=5, 
        marker='triangle', tools=['hover'])
Ant_bnds = gv.Shape.from_shapefile(
    str(ant_path), crs=ANT_proj).opts(
    projection=ANT_proj, width=700, height=700)
(Ant_bnds * radar_plt * core_plt)

#%% [markdown]
# ## Time series spectral analysis
# 
# Below are plots comparing the power spectral density of different cores and radar trace results.
# All data have been detrended and normalized using the time series mean and detrended standard deviation.
# I use Welch's method to estimate the periodograms (in my research this seemed to be the best choice due to increased signal to noise ratio, but I'm open to other suggestions).
# I have different PSD plots for in-situ cores, radar results, and synthetic white noise (for comparison).
# All results are for the consistent time span of 1979-2010.
#  
# %%

# Detrend and normalize data
core_array = signal.detrend(accum_core)
core_array = (
    (core_array - core_array.mean(axis=0)) 
    / core_array.std(axis=0))

fs_core, Pxx_core = signal.welch(
    core_array, axis=0, detrend=False)
# Pxx = Pxx.astype('complex128').real

core_results = pd.DataFrame(
    Pxx_core, index=fs_core, 
    columns=accum_core.columns)

ds_core = hv.Dataset(
    (np.arange(Pxx_core.shape[1]), 
    fs_core, Pxx_core), 
    ['Core', 'Frequency'], 'Power')

plt_core = ds_core.to(
    hv.Image, ['Core', 'Frequency'])
# plt_core = ds_core.to(
#     hv.Image, ['Core', 'Frequency']).hist()

tmp_range = np.arange(0,accum_core.shape[1])
x_ticks = [
    (int(_), accum_core.columns[_]) 
    for _ in tmp_range
]

plt_core = plt_core.opts(
    xticks=x_ticks, xrotation=90, 
    width=1000, height=600, colorbar=True)

#%% [markdown]
# Plot of power spectral density for firn/ice cores.
# Although the results are pretty varied, in general there seems to be more dominant periodicities near ~6 years, with additional peaks near 17 years and 2 years.
# Talos Dome has the strongest periodic signal at ~4 years.
#  
# %% Calculate power specta radar time series

accum_tsa = accum_grid

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
plt_accum = plt_accum.opts(
    width=1000, height=600, colorbar=True)

#%%[markdown]
# Plot showing the PSD for radar trace time series.
# Most of the power is concentrated in lower frequencies, with time series peaks ranging in periodicity from ~17 years to ~6 years, with the greatest concentration at ~13 year period.
# 
# **Question:** *Why am I getting results for periodicities greater than 1/2 the interval of interest? I would assume we could only measure repeat periodicity with period length <= 1/2 the total record interval, but maybe it's able to reconstruct results based off of partial periods? I also know Welch's method performs overlapping periodograms, so there is an averaging component to it that perhaps causes this? Related to this, I know we can get "spectral leakage" due to non-integer number of cycles in the record, but would this manifest as peaks past the 1/2 record length point? I suppose I would like some help/guidance on reviewing spectral analysis and interpreting the results...*
# 
# **Question:** *Additionally, I'm not sure the best way to compare the relative power in the spectral densities between the cores and the radar results (or even between one time series to the next of the same data type). My instinct is that these cannot be directly compared using the PSD magnitudes, but the relative strength of a peak in an individual time series compared to the corresponding lesser peaks within the same record can be used to infer how strongly a signal is periodic relative to other time series. I'm not actually sure about this and either need some direction or more research into it.*
# 
# %% Power spectra for white noise

# Compare results to white noise with the same mean and std of the detrended accum time series
ts_data = signal.detrend(accum_grid, axis=0)
df_test = np.random.normal(
    ts_data.mean(axis=0), ts_data.std(axis=0), 
    ts_data.shape)

df_norm = (
    (df_test - df_test.mean(axis=0)) 
    / df_test.std(axis=0))

fs_noise, Pxx_noise = signal.welch(df_norm, axis=0)

noise_results = pd.DataFrame(
    Pxx_noise, index=fs_noise, 
    columns=accum_grid.columns)

ds_noise = hv.Dataset(
    (np.arange(Pxx_noise.shape[1]), 
    fs_noise, Pxx_noise), 
    ['Trace_ID', 'Frequency'], 'Power')

noise_plt = ds_noise.to(
    hv.Image, ['Trace_ID', 'Frequency'])
noise_plt = noise_plt.opts(
    width=1000, height=600, colorbar=True)

#%%[markdown]
# Plot of PSD for white noise (as a comparison for results from the radar).
# The synthetic time series were generated from the detrended radar time series means and standard deviations, and were then normalized using the same methods as for the radar results.
# The major takeaway here is that the radar results express obvious patterns (specifically in the lower frequencies) that I believe are beyond what we would expect from random noise.
#  
# %% Violin plots of power spectra

# fig, axes = plt.figure(figsize=(6,9))
ax = plt.subplot(111)
ax.violinplot(
    Pxx_core.transpose(), showmedians=True)
ax.violinplot(
    Pxx_accum.transpose(), showmedians=True)

# %%

plt.rcParams.update({'font.size': 18})
P_fig, ax = plt.subplots()
P_fig.set_figheight(6)
P_fig.set_figwidth(10)
ax.plot(
    fs_core, np.median(Pxx_core, axis=1), color='blue', 
    label='Ice cores')
ax.plot(
    fs_accum, np.median(Pxx_accum, axis=1), color='red', 
    label='Radar traces')
ax.set_xlabel('Freqency (cycles/yr)')
ax.set_ylabel('Mean power density')
ax.legend(loc='best')

# %% 3D plot of core power spectra

import plotly.express as px
loc_tmp = core_locs.loc[accum_core.columns]
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
    color='Power', color_continuous_scale='viridis', 
    width=950, height=600)
fig.show()

#%% 3D plot of radar power spectra

E_accum = gdf_grid.geometry.centroid.x
N_accum = gdf_grid.geometry.centroid.y
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
    opacity=1, width=950, height=600)
fig.show()

# %% Plots of max periodicity estimate and max power

# Index of max spectral power
max_idx = accum_results.idxmax()

# Periodicity for max power
periods = (1/max_idx)

# Remove periodicity of inf or greater than 2/3 record range
periods[np.isinf(periods)] = 0

# Max power
psd_max = accum_results.max()


df_power = pd.DataFrame(
    {'Easting': E_accum, 'Northing': N_accum, 
    'Max period': periods, 'Max power': psd_max})

clipping = {
    'min': 'black', 'max': 'gray', 'NaN': 'gray'}

plt_max = hv.Points(
    df_power, kdims=['Easting', 'Northing'], 
    vdims=[
        hv.Dimension(
            'Max period', range=(0,25)), 
        'Max power']).opts(
        color='Max period', size='Max power')
plt_max.opts(
    cmap='viridis', width=950, height=600, 
    colorbar=True, alpha=0.50, tools=['hover'], 
    clipping_colors=clipping)

# %%[markdown]
# The above plot shows the radar trace time series analysis results, with the color representing the dominant period in each record and the size representing the relative power of that frequency.
# I've clipped the results to 2/3 of the total record duration (20 years) so grey circles represent periods longer than this.
#  
# %% Format generated plots

Pxx_plts = hv.Layout(
    plt_accum.opts(fontscale=2) 
    + noise_plt.opts(fontscale=2)
    + plt_core.opts(fontscale=2)).cols(2)

# %%

hv.save(Pxx_plts, ROOT_DIR.joinpath(
    'docs/Figures/Pxx_plts.png'))

P_fig.savefig(ROOT_DIR.joinpath(
    'docs/Figures/Pxx_fig.svg'))

# %%
