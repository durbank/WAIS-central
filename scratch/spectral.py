#%% [markdown]
# These results show some of the exploratory time series analysis I have been working on recently.
# These have mostly focused on spectral power and analysis to look for patterns in periodic signals within the radar SMB time series.
# These results focus on data from two flightlines (one in 2010 and one in 2011) covering the time period 1979-2010.
#  
# %%
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
The raw code for this Jupyter notebook is by default hidden for easier reading.
To toggle on/off the raw code, click <a href="javascript:code_toggle()">here</a>.''')

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
from bokeh.io import output_notebook
from shapely.geometry import Point
output_notebook()
hv.extension('bokeh')
gv.extension('bokeh')

# Set project root directory
ROOT_DIR = Path('/home/durbank/Documents/Research/Antarctica/WAIS-central/')

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

# Remove results for below QC data reliability
data_0 = data_raw[
    data_raw.Year > data_raw.QC_yr].sort_values(
        ['collect_time', 'Year'])

# # Subset data into QC flags 0, 1, and 2
# data_0 = data_raw[data_raw['QC_flag'] == 0]
# data_1 = data_raw[data_raw['QC_flag'] == 1]
# # data_2 = data_raw[data_raw['QC_flag'] == 2]

# # Remove data_1 values earlier than assigned QC yr, 
# # and recombine results with main data results
# data_1 = data_1[data_1.Year >= data_1.QC_yr]
# data_0 = data_0.append(data_1).sort_values(
#     ['collect_time', 'Year'])

#%%
# Format accumulation data
accum_long = format_PAIPR(
    data_0, start_yr=1975, end_yr=2010).drop(
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
# Subset core accum to same time period as radar
core_accum = core_ALL.transpose()
core_accum = core_accum[
    core_accum.index.isin(
        np.arange(1975,2010))].iloc[::-1]
core_accum = core_accum.iloc[
    :,(core_accum.isna().sum() <= 0).to_list()]
core_accum.columns.name = 'Core'

# Extract core locations for remaing cores
core_df_set = core_locs.loc[core_accum.columns]





# %%
## Autocorrelation analysis

core_acf = acf(core_accum)
core_acf.plot()






# %%
## Orientation plot showing location of data

radar_plt = gv.Points(
    accum_trace.sample(1000), crs=ANT_proj).opts(
        projection=ANT_proj, color='red')
core_plt = gv.Points(
    core_df_set, crs=ANT_proj, vdims=['Core']).opts(
        projection=ANT_proj, color='blue', size=5, 
        tools=['hover'])
Ant_bnds = gv.Shape.from_shapefile(
    str(ant_path), crs=ANT_proj).opts(
    projection=ANT_proj, width=500, height=500)
(Ant_bnds * radar_plt * core_plt)

# %%[markdown]
# The above plot shows the locations of data used.
# Only cores (blue) and radar results (red) covering the same consistent time period (1979-2010) are used.
# Note that some of the cores cover dispersed locations in East Antarctica, but are included here to show diversity in core results compared to the range in radar.
#  
# %%
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

tmp_range = np.arange(0,core_accum.shape[1])
x_ticks = [
    (int(_), core_accum.columns[_]) 
    for _ in tmp_range
]
plt_core.opts(
    xticks=x_ticks, xrotation=90, 
    width=950, height=600, colorbar=True)

#%% [markdown]
# Plot of power spectral density for firn/ice cores.
# Although the results are pretty varied, in general there seems to be more dominant periodicities near ~6 years, with additional peaks near 17 years and 2 years.
# Talos Dome has the strongest periodic signal at ~4 years.
#   
# %%
## Calculate power specta for accum time series (both radar results and cores)

accum_tsa = accum.iloc[:,::40]
# accum_tsa = accum

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
plt_accum.opts(width=950, height=600, colorbar=True)

#%%[markdown]
# Plot showing the PSD for radar trace time series.
# To aid in plotting, only the results for traces every 1 km along the flightline are shown.
# Most of the power is concentrated in lower frequencies, with time series peaks ranging in periodicity from ~17 years to ~6 years, with the greatest concentration at ~13 year period.
# 
# **Question:** *Why am I getting results for periodicities greater than 1/2 the interval of interest? I would assume we could only measure repeat periodicity with period length <= 1/2 the total record interval, but maybe it's able to reconstruct results based off of partial periods? I also know Welch's method performs overlapping periodograms, so there is an averaging component to it that perhaps causes this? Related to this, I know we can get "spectral leakage" due to non-integer number of cycles in the record, but would this manifest as peaks past the 1/2 record length point? I suppose I would like some help/guidance on reviewing spectral analysis and interpreting the results...*
# '
# **Question:** *Additionally, I'm not sure the best way to compare the relative power in the spectral densities between the cores and the radar results (or even between one time series to the next of the same data type). My instinct is that these cannot be directly compared using the PSD magnitudes, but the relative strength of a peak in an individual time series compared to the corresponding lesser peaks within the same record can be used to infer how strongly a signal is periodic relative to other time series. I'm not actually sure about this and either need some direction or more research into it.*
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
    hv.Image, ['Trace_ID', 'Frequency'])
noise_plt.opts(width=950, height=600, colorbar=True)

#%%[markdown]
# Plot of PSD for white noise (as a comparison for results from the radar).
# The synthetic time series were generated from the detrended radar time series means and standard deviations, and were then normalized using the same methods as for the radar results.
# The major takeaway here is that the radar results express obvious patterns that I believe are beyond what we would expect from random noise.
#  
# %%
ax = plt.subplot(111)
plt.plot(
    fs_core, Pxx_core.mean(axis=1), color='blue', 
    label='Ice cores')
plt.plot(
    fs_accum, Pxx_accum.mean(axis=1), color='red', 
    label='Radar traces')
plt.xlabel('Freqency (cycles/yr)')
plt.ylabel('Mean power density')
plt.legend(loc='best')
#%% [markdown]
# ## Spatial plots of PSD results
# 
# Below are plots of core and radar time series spatially plotted (in Polar Stereographic projection).
# This is still something of a work in progress, but should give some idea as to how spatially coherent/divergent the time series results are.
#  
# %%
# 3D plot of time series analysis
import plotly.express as px
loc_tmp = core_locs.loc[core_accum.columns]
E_core = loc_tmp.geometry.x
N_core = loc_tmp.geometry.y

# tmp_core = fs_core
# tmp_core[tmp_core <= 0] = 1
# P_core = 1 / tmp_core
# df_core = pd.DataFrame(
#     {'Easting': E_core.repeat(Pxx_core.shape[0]), 
#     'Northing': N_core.repeat(Pxx_core.shape[0]), 
#     'Period': np.tile(P_core, E_core.shape[0]), 
#     'Power': Pxx_core.reshape(
#         Pxx_core.size, 1).squeeze()})
# fig = px.scatter_3d(
#     df_core, 
#     x='Easting', y='Northing', z='Period', 
#     color='Power', color_continuous_scale='viridis', 
#     width=950, height=600)
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

# %%[markdown]
# This shows the core spectral results spatially.
#  
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
    opacity=1, width=950, height=600)
fig.show()

# %%[markdown]
# This shows the radar spectral results spatially (again subsetted to every 1 km to aid in viewing).
#  
#%%
## Plots of max periodicity estimate and max power
# Index of max spectral power
max_idx = accum_results.idxmax()

# Periodicity for max power
periods = (1/max_idx)

# Remove periodicity of inf or greater than 2/3 record range
periods[np.isinf(periods)] = 0

# Max power
psd_max = accum_results.max()


loc_tmp = accum_trace.loc[accum_tsa.columns]
E_accum = loc_tmp.geometry.x
N_accum = loc_tmp.geometry.y

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