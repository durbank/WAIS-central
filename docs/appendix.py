# %%[markdown]
# # Appendix
# 
# This notebook provides scripts, code, and notes for the appendices of the article.
# It includes annual smb comparisons for manually-drawn layers vs PAIPR-generated results.
# It further includes repeatability results comparing annual smb results and trends using PAIPR for the 2011 vs 2016 repeat flights.
# Finally, it also compares the 2011 vs 2016 repeatability of PAIPR to that of the manually-drawn smb results. 
# 
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
import seaborn as sns

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)

# Set project root directory
ROOT_DIR = ROOT_DIR = Path(__file__).parents[1]

# Import custom project functions
import sys
SRC_DIR = ROOT_DIR.joinpath('src')
sys.path.append(str(SRC_DIR))
from my_functions import *

# %% Import PAIPR-generated data
# Import 20111109 results
dir1 = ROOT_DIR.joinpath('data/PAIPR-outputs/20111109/')
data_raw = import_PAIPR(dir1)
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)
data_2011 = format_PAIPR(
    data_0, start_yr=1990, end_yr=2010).drop(
    'elev', axis=1)
a2011_ALL = data_2011.pivot(
    index='Year', columns='trace_ID', values='accum')
std2011_ALL = data_2011.pivot(
    index='Year', columns='trace_ID', values='std')

# Import 20161109 results
dir2 = ROOT_DIR.joinpath('data/PAIPR-outputs/20161109/')
data_raw = import_PAIPR(dir2)
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)
data_2016 = format_PAIPR(
    data_0, start_yr=1990, end_yr=2010).drop(
    'elev', axis=1)
a2016_ALL = data_2016.pivot(
    index='Year', columns='trace_ID', values='accum')
std2016_ALL = data_2016.pivot(
    index='Year', columns='trace_ID', values='std')

# Create gdf of mean results for each trace and 
# transform to Antarctic Polar Stereographic
gdf_2011 = long2gdf(data_2011)
gdf_2011.to_crs(epsg=3031, inplace=True)
gdf_2016 = long2gdf(data_2016)
gdf_2016.to_crs(epsg=3031, inplace=True)

# %%

trend_2011, _, lb_2011, ub_2011 = trend_bs(
    a2011_ALL, 1000, df_err=std2011_ALL)
gdf_2011['accum'] = a2011_ALL.mean().values
gdf_2011['trend'] = trend_2011 / gdf_2011['accum']
gdf_2011['t_lb'] = lb_2011 / gdf_2011['accum']
gdf_2011['t_ub'] = ub_2011 / gdf_2011['accum']

trend_2016, _, lb_2016, ub_2016 = trend_bs(
    a2016_ALL, 1000, df_err=std2016_ALL)
gdf_2016['accum'] = a2016_ALL.mean().values
gdf_2016['trend'] = trend_2016 / gdf_2016['accum']
gdf_2016['t_lb'] = lb_2016 / gdf_2016['accum']
gdf_2016['t_ub'] = ub_2016 / gdf_2016['accum']

#%%
plt_accum2011 = gv.Points(
    gdf_2011, crs=ANT_proj, vdims=['accum']).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, 
        bgcolor='silver', tools=['hover'], 
        width=600, height=400)
plt_accum2016 = gv.Points(
    gdf_2016, crs=ANT_proj, vdims=['accum']).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, 
        bgcolor='silver', tools=['hover'], 
        width=600, height=400)
plt_accum2011 + plt_accum2016

# %%
plt_trend2011 = gv.Points(
    gdf_2011, crs=ANT_proj, 
    vdims=['trend','t_lb','t_ub']).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm_r', symmetric = True, 
        colorbar=True, bgcolor='silver', 
        tools=['hover'], width=600, height=400)
plt_trend2016 = gv.Points(
    gdf_2016, crs=ANT_proj, 
    vdims=['trend','t_lb','t_ub']).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm_r', colorbar=True, 
        symmetric=True, bgcolor='silver', 
        tools=['hover'], width=600, height=400)
plt_trend2011 + plt_trend2016

# %% Import manual results
# Get list of 2011 manual files
# list_2011 = [file for file in Path('data/smb_manual').glob('*20111109.csv')]
dir_0 = ROOT_DIR.joinpath('data/smb_manual/20111109/')
data_0 = import_PAIPR(dir_0)
man_2011 = format_PAIPR(
    data_0, start_yr=1990, end_yr=2010).drop(
    'elev', axis=1)
man2011_ALL = man_2011.pivot(
    index='Year', columns='trace_ID', values='accum')
manSTD_2011_ALL = man_2011.pivot(
    index='Year', columns='trace_ID', values='std')

# Create gdf of mean results for each trace and 
# transform to Antarctic Polar Stereographic
gdf_man2011 = long2gdf(man_2011)
gdf_man2011.to_crs(epsg=3031, inplace=True)

# Perform same for 2016 manual results
dir_0 = ROOT_DIR.joinpath('data/smb_manual/20161109/')
data_0 = import_PAIPR(dir_0)
man_2016 = format_PAIPR(
    data_0, start_yr=1990, end_yr=2010).drop(
    'elev', axis=1)
man2016_ALL = man_2016.pivot(
    index='Year', columns='trace_ID', values='accum')
manSTD_2016_ALL = man_2016.pivot(
    index='Year', columns='trace_ID', values='std')

# Create gdf of mean results for each trace and 
# transform to Antarctic Polar Stereographic
gdf_man2016 = long2gdf(man_2016)
gdf_man2016.to_crs(epsg=3031, inplace=True)

# %%
df_dist = nearest_neighbor(
    gdf_man2011, gdf_2011, return_dist=True)
idx_2011 = df_dist['distance'] <= 250
dist_overlap1 = df_dist[idx_2011]

# Create numpy arrays for relevant results
man2011_accum = man2011_ALL.iloc[
    :,dist_overlap1.index]
man2011_std = manSTD_2011_ALL.iloc[
    :,dist_overlap1.index]
Maccum_2011 = a2011_ALL.iloc[
    :,dist_overlap1['trace_ID']]
Mstd_2011 = std2011_ALL.iloc[
    :,dist_overlap1['trace_ID']]

# Create new gdf of subsetted results
gdf_traces2011 = gpd.GeoDataFrame(
    {'ID_man': dist_overlap1.index.values, 
    'ID_PAIPR': dist_overlap1['trace_ID'].values, 
    'accum_man': man2011_accum.mean(axis=0).values, 
    'accum_PAIPR': Maccum_2011.mean(axis=0).values},
    geometry=dist_overlap1.geometry.values)


# Calculate residuals (as % bias of mean accumulation)
res_2011 = (
    (Maccum_2011.values - man2011_accum.values) 
    / np.mean(
        [Maccum_2011.mean(axis=0).values, 
        man2011_accum.mean(axis=0).values], axis=0))

sns.kdeplot(res_2011.reshape(res_2011.size))

print(
    f"The mean bias between manually-traced and "
    f"PAIPR-derived accumulation for 2011 is "
    f"{res_2011.mean()*100:.2f}% "
    f"with a RMSE of {res_2011.std()*100:.2f}% "
    f"and a RMSE in mean accumulation of {res_2011.mean(axis=0).std()*100:.2f}%"
)

print(
    f"The mean standard deviations of the annual "
    f"accumulation estimates are "
    f"{(man2011_std/man2011_accum).values.mean()*100:.2f}% " 
    f"for 2011 manual layers and "
    f"{(Mstd_2011/Maccum_2011).values.mean()*100:.2f}% "
    f"for 2011 PAIPR layers."
)

# %%
# Assign flight chunk class based on location
chunk_centers = gpd.GeoDataFrame(
    ['PIG', 'SEAT2010-6', 'SEAT2010-5', 
    'SEAT2010-4', 'high-accum'],
    geometry=gpd.points_from_xy(
        [-1.297E6, -1.024E6, -1.093E6, 
            -1.159E6, -1.263E6], 
        [-1.409E5, -4.639E5, -4.639E5, 
            -4.640E5, -4.868E5]), 
    crs="EPSG:3031")

PIG_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(chunk_centers.geometry.x[0], gdf_traces2011.shape[0]), 
        np.repeat(chunk_centers.geometry.y[0], gdf_traces2011.shape[0])), 
        crs="EPSG:3031")
    .distance(gdf_traces2011.reset_index())) <= 30000).values

SEAT6_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(chunk_centers.geometry.x[1], gdf_traces2011.shape[0]), 
        np.repeat(chunk_centers.geometry.y[1], gdf_traces2011.shape[0])), 
        crs="EPSG:3031")
    .distance(gdf_traces2011.reset_index())) <= 30000).values

SEAT5_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(chunk_centers.geometry.x[2], gdf_traces2011.shape[0]), 
        np.repeat(chunk_centers.geometry.y[2], gdf_traces2011.shape[0])), 
        crs="EPSG:3031")
    .distance(gdf_traces2011.reset_index())) <= 30000).values

SEAT4_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(chunk_centers.geometry.x[3], gdf_traces2011.shape[0]), 
        np.repeat(chunk_centers.geometry.y[3], gdf_traces2011.shape[0])), 
        crs="EPSG:3031")
    .distance(gdf_traces2011.reset_index())) <= 30000).values

high_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(chunk_centers.geometry.x[4], gdf_traces2011.shape[0]), 
        np.repeat(chunk_centers.geometry.y[4], gdf_traces2011.shape[0])), 
        crs="EPSG:3031")
    .distance(gdf_traces2011.reset_index())) <= 30000).values

gdf_traces2011['Site'] = np.repeat(
    'Null', gdf_traces2011.shape[0])
gdf_traces2011['Site'][PIG_idx] = 'PIG'
gdf_traces2011['Site'][SEAT6_idx] = 'SEAT2010-6'
gdf_traces2011['Site'][SEAT5_idx] = 'SEAT2010-5'
gdf_traces2011['Site'][SEAT4_idx] = 'SEAT2010-4'
gdf_traces2011['Site'][high_idx] = 'high-accum'

# %%

for idx in [PIG_idx, SEAT4_idx, SEAT5_idx, SEAT6_idx, high_idx]:
    df_paipr = Maccum_2011.iloc[:,idx]
    df_manual = man2011_accum.iloc[:,idx]
    fig, ax = plt.subplots()
    
    df_paipr.mean(axis=1).plot(ax=ax, color='red', 
        linewidth=2, label='2011 PAIPR')
    (df_paipr.mean(axis=1)+df_paipr.std(axis=1)).plot(
        ax=ax, color='red', linestyle='--', 
        label='__nolegend__')
    (df_paipr.mean(axis=1)-df_paipr.std(axis=1)).plot(
        ax=ax, color='red', linestyle='--', 
        label='__nolegend__')

    df_manual.mean(axis=1).plot(
        ax=ax, color='blue', linewidth=2, 
        label='2011 manual')
    (df_manual.mean(axis=1)+df_manual.std(axis=1)).plot(
        ax=ax, color='blue', linestyle='--', 
        label='__nolegend__')
    (df_manual.mean(axis=1)-df_manual.std(axis=1)).plot(
        ax=ax, color='blue', linestyle='--', 
        label='__nolegend__')
    ax.legend()
    fig.suptitle('Test Title')
    fig.show()

# %%
# Create dataframes for scatter plots
df_2011 = pd.DataFrame(
    {'Site': np.tile(
        gdf_traces2011['Site'], 
        Maccum_2011.shape[0]), 
    'Year': np.reshape(
        np.repeat(
            a2011_ALL.index, 
            Maccum_2011.shape[1]), 
        Maccum_2011.size), 
    'accum_man': np.reshape(
        man2011_accum.values, man2011_accum.size),  
    'accum_paipr': np.reshape(
        Maccum_2011.values, Maccum_2011.size)})

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=df_2011, 
    kdims=['accum_man', 'accum_paipr'], 
    vdims=['Year']).groupby('Year')
(
    one_to_one.opts(color='black') 
    * scatt_yr.opts(
        xlim=(100,750), ylim=(100,750), 
        xlabel='Manual accum (mm/yr)', 
        ylabel='PAIPR accum (mm/yr)'))

# %%
one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=df_2011, 
    kdims=['accum_man', 'accum_paipr'], 
    vdims=['Site']).groupby('Site')
(
    one_to_one.opts(color='black') 
    * scatt_yr.opts(
        xlim=(100,750), ylim=(100,750), 
        xlabel='Manual accum (mm/yr)', 
        ylabel='PAIPR accum (mm/yr)'))

# %% Trend analysis
trend_man, intcpt_man, lb_man, ub_man = trend_bs(
    man2011_accum, 1000, df_err=man2011_std)

trend_paipr,intcpt_paipr,lb_paipr,ub_paipr = trend_bs(
    Maccum_2011, 1000, df_err=Mstd_2011)



# %%






# %%
df_dist = nearest_neighbor(
    gdf_man2016, gdf_2016, return_dist=True)
idx_2016 = df_dist['distance'] <= 250
dist_overlap2 = df_dist[idx_2011]

# Create numpy arrays for relevant results
man2016_accum = man2016_ALL.iloc[
    :,dist_overlap2.index]
man2016_std = manSTD_2016_ALL.iloc[
    :,dist_overlap2.index]
Maccum_2016 = a2016_ALL.iloc[
    :,dist_overlap2['trace_ID']]
Mstd_2016 = std2016_ALL.iloc[
    :,dist_overlap2['trace_ID']]

# Create new gdf of subsetted results
gdf_traces2016 = gpd.GeoDataFrame(
    {'ID_man': dist_overlap2.index.values, 
    'ID_PAIPR': dist_overlap2['trace_ID'].values, 
    'accum_man': man2016_accum.mean(axis=0).values, 
    'accum_PAIPR': Maccum_2016.mean(axis=0).values},
    geometry=dist_overlap2.geometry.values)


# Calculate residuals (as % bias of mean accumulation)
res_2016 = (
    (Maccum_2016.values - man2016_accum.values) 
    / np.mean(
        [Maccum_2016.mean(axis=0), 
        man2016_accum.mean(axis=0)], axis=0))

sns.kdeplot(res_2016.reshape(res_2016.size))

print(
    f"The mean bias between manually-traced and "
    f"PAIPR-derived accumulation for 2016 is "
    f"{res_2016.mean()*100:.2f}% "
    f"with a RMSE of {res_2016.std()*100:.2f}% "
    f"and a RMSE in mean accumulation of {res_2016.mean(axis=0).std()*100:.2f}%"
)

print(
    f"The mean standard deviations of the annual "
    f"accumulation estimates are "
    f"{(man2016_std/man2016_accum).values.mean()*100:.2f}% " 
    f"for 2016 manual layers and "
    f"{(Mstd_2016/Maccum_2016).values.mean()*100:.2f}% "
    f"for 2016 PAIPR layers."
)

# %% Repeatability tests

df_dist = nearest_neighbor(
    gdf_2011, gdf_2016, return_dist=True)
idx_paipr = df_dist['distance'] <= 250
dist_overlap = df_dist[idx_paipr]

# Create numpy arrays for relevant results
accum_2011 = a2011_ALL.iloc[
    :,dist_overlap.index]
std_2011 = std2011_ALL.iloc[
    :,dist_overlap.index]
accum_2016 = a2016_ALL.iloc[
    :,dist_overlap['trace_ID']]
std_2016 = std2016_ALL.iloc[
    :,dist_overlap['trace_ID']]

# # Diagnostic plots
# for idx in [0, 100, 250, 500, 1000, 1500]:
#     fig, ax = plt.subplots()
#     accum_2011.iloc[:,idx].plot(
#         ax=ax, color='blue', linewidth=2, 
#         label='2011 flight')
#     (accum_2011.iloc[:,idx]+std_2011.iloc[:,idx]).plot(
#         ax=ax, color='blue', linestyle='--', 
#         label='__nolegend__')
#     (accum_2011.iloc[:,idx]-std_2011.iloc[:,idx]).plot(
#         ax=ax, color='blue', linestyle='--', 
#         label='__nolegend__')
#     accum_2016.iloc[:,idx].plot(
#         ax=ax, color='red', linewidth=2, 
#         label='2016 flight')
#     (accum_2016.iloc[:,idx]+std_2016.iloc[:,idx]).plot(
#         ax=ax, color='red', linestyle='--', 
#         label='__nolegend__')
#     (accum_2016.iloc[:,idx]-std_2016.iloc[:,idx]).plot(
#         ax=ax, color='red', linestyle='--', 
#         label='__nolegend__')
#     ax.legend()
#     fig.show()

# %%
# Create new gdf of subsetted results
gdf_PAIPR = gpd.GeoDataFrame(
    {'ID_2011': dist_overlap.index.values, 
    'ID_2016': dist_overlap['trace_ID'].values, 
    'accum_2011': 
        accum_2011.mean(axis=0).values, 
    'accum_2016': 
        accum_2016.mean(axis=0).values},
    geometry=dist_overlap.geometry.values)


# Calculate residuals (as % bias of mean accumulation)
res_PAIPR = (
    (accum_2016.values - accum_2011.values) 
    / np.mean(
        [accum_2016.mean(axis=0).values, 
        accum_2011.mean(axis=0).values], axis=0))

sns.kdeplot(res_PAIPR.reshape(res_PAIPR.size))

print(
    f"The mean bias between PAIPR-derived results "
    f"between 2016 and 2011 flights is "
    f"{res_PAIPR.mean()*100:.2f}% "
    f"with a RMSE of {res_PAIPR.std()*100:.2f}% "
    f"and a RMSE in mean accumulation of {res_PAIPR.mean(axis=0).std()*100:.2f}%"
)

print(
    f"The mean standard deviations of the annual "
    f"accumulation estimates are "
    f"{(std_2011.values/accum_2011.values).mean()*100:.2f}% " 
    f"for the 2011 flight and "
    f"{(std_2016.values/accum_2016.values).mean()*100:.2f}% "
    f"for the 2016 flight."
)
# %%
trend_2011, intcpt, lb_2011, ub_2011 = trend_bs(
    accum_2011, 1000, df_err=std_2011)

trend_2016, intcpt, lb_2016, ub_2016 = trend_bs(
    accum_2016, 1000, df_err=std_2016)

accum_bar = gdf_PAIPR[[
    'accum_2011', 'accum_2016']].mean(
        axis=1).values
gdf_PAIPR['trend_2011'] = trend_2011 / accum_bar
gdf_PAIPR['t2011_lb'] = lb_2011 / accum_bar
gdf_PAIPR['t2011_ub'] = ub_2011 / accum_bar
gdf_PAIPR['trend_2016'] = trend_2016 / accum_bar
gdf_PAIPR['t2016_lb'] = lb_2016 / accum_bar
gdf_PAIPR['t2016_ub'] = ub_2016 / accum_bar
gdf_PAIPR['trend_res'] = trend_2016 - trend_2011

plt.plot([-0.03,0.04], [-0.03,0.04], color='black')
plt.scatter(gdf_PAIPR['trend_2011'], gdf_PAIPR['trend_2016'])
plt.show()

plt.plot([-20,5], [-20,5], color='black')
plt.scatter(
    gdf_PAIPR['trend_2011'] * accum_bar, 
    gdf_PAIPR['trend_2016'] * accum_bar)
plt.show()

# %%
sns.kdeplot(gdf_PAIPR['trend_res'])

print(
    f"The mean trend bias between 2016 and 2011 "
    f"PAIPR-derived estimates is "
    f"{gdf_PAIPR['trend_res'].mean():.2f} mm/yr "
    f"(~{100*gdf_PAIPR['trend_res'].mean()/gdf_PAIPR['accum_2011'].mean():.2f}% of the mean accum)")
# print(
#     f"The mean error (in %/yr) for trend values is "
#     f"{100*((gdf_traces['t2011_ub']-gdf_traces['t2011_lb'])/2).mean():.2f}% "
#     f"for the 2011 PAIPR estimates and "
#     f"{100*((gdf_traces['t2016_ub']-gdf_traces['t2016_lb'])/2).mean():.2f}% "
#     f"for the 2016 manual estimates")
# %%

plt_trend = gv.Points(
    data=gdf_PAIPR, 
    vdims=['trend_res', 'trend_2011', 'trend_2016'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trend_res', 
        cmap='coolwarm', symmetric=True, colorbar=True, bgcolor='silver', 
        tools=['hover'], width=700, height=700)
plt_trend

# %%






# # %% Import manual results
# # Get list of 2011 manual files
# # list_2011 = [file for file in Path('data/smb_manual').glob('*20111109.csv')]
# dir_0 = ROOT_DIR.joinpath('data/smb_manual/20111109/')
# data_0 = import_PAIPR(dir_0)
# man_2011 = format_PAIPR(
#     data_0, start_yr=1987, end_yr=2010).drop(
#     'elev', axis=1)
# man2011_ALL = man_2011.pivot(
#     index='Year', columns='trace_ID', values='accum')
# manSTD_2011_ALL = man_2011.pivot(
#     index='Year', columns='trace_ID', values='std')

# # Create gdf of mean results for each trace and 
# # transform to Antarctic Polar Stereographic
# gdf_man2011 = long2gdf(man_2011)
# gdf_man2011.to_crs(epsg=3031, inplace=True)

# # Perform same for 2016 manual results
# dir_0 = ROOT_DIR.joinpath('data/smb_manual/20161109/')
# data_0 = import_PAIPR(dir_0)
# man_2016 = format_PAIPR(
#     data_0, start_yr=1987, end_yr=2010).drop(
#     'elev', axis=1)
# man2016_ALL = man_2016.pivot(
#     index='Year', columns='trace_ID', values='accum')
# manSTD_2016_ALL = man_2016.pivot(
#     index='Year', columns='trace_ID', values='std')

# # Create gdf of mean results for each trace and 
# # transform to Antarctic Polar Stereographic
# gdf_man2016 = long2gdf(man_2016)
# gdf_man2016.to_crs(epsg=3031, inplace=True)

# # %% Subset manual results to overlapping sections
# df_dist = nearest_neighbor(
#     gdf_man2016, gdf_man2011, return_dist=True)
# idx_2016 = df_dist['distance'] <= 250
# dist_overlap = df_dist[idx_2016]

# # Create numpy arrays for relevant results
# accum_man2011 = (
#     man2011_ALL.iloc[:,dist_overlap['trace_ID']].to_numpy())
# std_man2011 = (
#     manSTD_2011_ALL.iloc[:,dist_overlap['trace_ID']].to_numpy())
# accum_man2016 = (
#     man2016_ALL.iloc[:,dist_overlap.index].to_numpy())
# std_man2016 = (
#     manSTD_2011_ALL.iloc[:,dist_overlap.index].to_numpy())

# # Create new gdf of subsetted results
# gdf_traces = gpd.GeoDataFrame(
#     {'ID_2011': dist_overlap['trace_ID'], 
#     'ID_2016': dist_overlap.index, 
#     'accum_man2011': accum_man2011.mean(axis=0), 
#     'accum_man2016': accum_man2016.mean(axis=0)},
#     geometry=dist_overlap.geometry.values)

# # Assign flight chunk class based on location
# chunk_centers = gpd.GeoDataFrame(
#     ['mid-accum', 'SEAT2010-4', 'high-accum'],
#     geometry=gpd.points_from_xy(
#         [-1.177E6, -1.159E6, -1.263E6], 
#         [-2.898E5, -4.640E5, -4.868E5]), 
#     crs="EPSG:3031")

# mid_accum_idx = ((
#     gpd.GeoSeries(gpd.points_from_xy(
#         np.repeat(chunk_centers.geometry.x[0], gdf_traces.shape[0]), 
#         np.repeat(chunk_centers.geometry.y[0], gdf_traces.shape[0])), 
#         crs="EPSG:3031")
#     .distance(gdf_traces.reset_index())) <= 30000).values

# SEAT4_idx = ((
#     gpd.GeoSeries(gpd.points_from_xy(
#         np.repeat(chunk_centers.geometry.x[1], gdf_traces.shape[0]), 
#         np.repeat(chunk_centers.geometry.y[1], gdf_traces.shape[0])), 
#         crs="EPSG:3031")
#     .distance(gdf_traces.reset_index())) <= 30000).values
        
# high_accum_idx = ((
#     gpd.GeoSeries(gpd.points_from_xy(
#         np.repeat(chunk_centers.geometry.x[2], gdf_traces.shape[0]), 
#         np.repeat(chunk_centers.geometry.y[2], gdf_traces.shape[0])), 
#         crs="EPSG:3031")
#     .distance(gdf_traces.reset_index())) <= 30000).values

# gdf_traces['Site'] = np.repeat('Null', gdf_traces.shape[0])
# gdf_traces['Site'][mid_accum_idx] = 'mid-accum'
# gdf_traces['Site'][SEAT4_idx] = 'SEAT2010-4'
# gdf_traces['Site'][high_accum_idx] = 'high-accum'

# # %%
# # Calculate residuals (as % bias of mean accumulation)
# man_res = (
#     (accum_man2016 - accum_man2011) 
#     / np.mean(
#         [accum_man2011.mean(axis=0), 
#         accum_man2016.mean(axis=0)], axis=0))

# sns.kdeplot(man_res.reshape(man_res.size))

# print(
#     f"The mean bias in manually-derived annual accumulation "
#     f"for 2016 vs. 2011 flights is "
#     f"{man_res.mean()*100:.2f}% "
#     f"with a RMSE of {man_res.std()*100:.2f}% "
#     f"and a RMSE in mean accumulation of {man_res.mean(axis=0).std()*100:.2f}%"
# )

# print(
#     f"The mean standard deviations of the annual accumulation "
#     f"estimates are "
#     f"{(std_man2011/accum_man2011).mean()*100:.2f}% for " f"2011 and "
#     f"{(std_man2016/accum_man2016).mean()*100:.2f}% for 2016."
# )

# # %%

# # Create dataframes for scatter plots
# accum_df = pd.DataFrame(
#     {'Trace': np.tile(
#         np.arange(0,accum_man2011.shape[1]), 
#         accum_man2011.shape[0]), 
#     'Site': np.tile(
#         gdf_traces['Site'], accum_man2011.shape[0]), 
#     'Year': np.reshape(
#         np.repeat(man2011_ALL.index, accum_man2011.shape[1]), 
#         accum_man2011.size), 
#     'accum_2011': 
#         np.reshape(accum_man2011, accum_man2011.size), 
#     'std_2011': np.reshape(std_man2011, std_man2011.size), 
#     'accum_2016': 
#         np.reshape(accum_man2016, accum_man2016.size), 
#     'std_2016': np.reshape(std_man2016, std_man2016.size)})

# one_to_one = hv.Curve(
#     data=pd.DataFrame(
#         {'x':[100,750], 'y':[100,750]}))
# scatt_yr = hv.Points(
#     data=accum_df, 
#     kdims=['accum_2011', 'accum_2016'], 
#     vdims=['Year']).groupby('Year')
# (
#     one_to_one.opts(color='black') 
#     * scatt_yr.opts(
#         xlim=(100,750), ylim=(100,750), 
#         xlabel='2011 flight (mm/yr)', 
#         ylabel='2016 flight (mm/yr)'))

# # %%

# one_to_one = hv.Curve(
#     data=pd.DataFrame(
#         {'x':[100,750], 'y':[100,750]}))
# scatt_yr = hv.Points(
#     data=accum_df, 
#     kdims=['accum_2011', 'accum_2016'], 
#     vdims=['Site']).groupby('Site')
# (
#     one_to_one.opts(color='black') 
#     * scatt_yr.opts(
#         xlim=(100,750), ylim=(100,750), 
#         xlabel='2011 flight (mm/yr)', 
#         ylabel='2016 flight (mm/yr)'))

# # %% Mean time series for manual mid-accum

# man_mid2011 = pd.DataFrame(
#     accum_man2011[:,mid_accum_idx], 
#     index=man2011_ALL.index)
# man_mid2016 = pd.DataFrame(
#     accum_man2016[:,mid_accum_idx], 
#     index=man2016_ALL.index)

# man_mid2011.mean(axis=1).plot(
#     color='red', linewidth=2, 
#     label='2011 flight')
# (
#     man_mid2011.mean(axis=1) 
#     + man_mid2011.std(axis=1)).plot(
#         color='red', linestyle='--', 
#         label='__nolegend__')
# (
#     man_mid2011.mean(axis=1) 
#     - man_mid2011.std(axis=1)).plot(
#         color='red', linestyle='--', 
#         label='__nolegend__')
# man_mid2016.mean(axis=1).plot(
#     color='blue', linewidth=2, 
#     label='2016 flight')
# (
#     man_mid2016.mean(axis=1) 
#     + man_mid2016.std(axis=1)).plot(
#         color='blue', linestyle='--', 
#         label='__nolegend__')
# (
#     man_mid2016.mean(axis=1) 
#     - man_mid2016.std(axis=1)).plot(
#         color='blue', linestyle='--', 
#         label='__nolegend__')
# plt.legend()
# # %% Mean time series for manual SEAT2010-4

# man_SEAT2011 = pd.DataFrame(
#     accum_man2011[:,SEAT4_idx], 
#     index=man2011_ALL.index)
# man_SEAT2016 = pd.DataFrame(
#     accum_man2016[:,SEAT4_idx], 
#     index=man2016_ALL.index)

# man_SEAT2011.mean(axis=1).plot(
#     color='red', linewidth=2, 
#     label='2011 flight')
# (
#     man_SEAT2011.mean(axis=1) 
#     + man_SEAT2011.std(axis=1)).plot(
#         color='red', linestyle='--', 
#         label='__nolegend__')
# (
#     man_SEAT2011.mean(axis=1) 
#     - man_SEAT2011.std(axis=1)).plot(
#         color='red', linestyle='--', 
#         label='__nolegend__')

# man_SEAT2016.mean(axis=1).plot(
#     color='blue', linewidth=2, 
#     label='2016 flight')
# (
#     man_SEAT2016.mean(axis=1) 
#     + man_SEAT2016.std(axis=1)).plot(
#         color='blue', linestyle='--', 
#         label='__nolegend__')
# (
#     man_SEAT2016.mean(axis=1) 
#     - man_SEAT2016.std(axis=1)).plot(
#         color='blue', linestyle='--', 
#         label='__nolegend__')
# plt.legend()

# # %% Mean time series for manual high-accum

# man_high2011 = pd.DataFrame(
#     accum_man2011[:,high_accum_idx], 
#     index=man2011_ALL.index)
# man_high2016 = pd.DataFrame(
#     accum_man2016[:,high_accum_idx], 
#     index=man2016_ALL.index)

# man_high2011.mean(axis=1).plot(
#     color='red', linewidth=2, 
#     label='2011 flight')
# (
#     man_high2011.mean(axis=1) 
#     + man_high2011.std(axis=1)).plot(
#         color='red', linestyle='--', 
#         label='__nolegend__')
# (
#     man_high2011.mean(axis=1) 
#     - man_high2011.std(axis=1)).plot(
#         color='red', linestyle='--', 
#         label='__nolegend__')

# man_high2016.mean(axis=1).plot(
#     color='blue', linewidth=2, 
#     label='2016 flight')
# (
#     man_high2016.mean(axis=1) 
#     + man_high2016.std(axis=1)).plot(
#         color='blue', linestyle='--', 
#         label='__nolegend__')
# (
#     man_high2016.mean(axis=1) 
#     - man_high2016.std(axis=1)).plot(
#         color='blue', linestyle='--', 
#         label='__nolegend__')
# plt.legend()

# # %%
# T_man2011, intcpt, lb_man2011, ub_man2011 = trend_bs(
#     pd.DataFrame(
#         accum_man2011, index=man2011_ALL.index), 
#     1000, 
#     df_err=pd.DataFrame(
#         std_man2011, index=man2011_ALL.index))

# T_man2016, intcpt, lb_man2016, ub_man2016 = trend_bs(
#     pd.DataFrame(
#         accum_man2016, index=man2016_ALL.index), 
#     1000, 
#     df_err=pd.DataFrame(
#         std_man2016, index=man2016_ALL.index))

# accum_bar = gdf_traces[
#     ['accum_man2011', 'accum_man2016']].mean(
#         axis=1).values
# gdf_traces['trend_2011'] = T_man2011 / accum_bar
# gdf_traces['t2011_lb'] = lb_man2011 / accum_bar
# gdf_traces['t2011_ub'] = ub_man2011 / accum_bar
# gdf_traces['trend_2016'] = T_man2016 / accum_bar
# gdf_traces['t2016_lb'] = lb_man2016 / accum_bar
# gdf_traces['t2016_ub'] = ub_man2016 / accum_bar
# gdf_traces['trend_res'] = (
#     T_man2016 - T_man2011) / accum_bar

# one_to_one = hv.Curve(
#     data=pd.DataFrame(
#         {'x':[-0.035,0.01], 'y':[-0.035,0.01]}))
# scatt_yr = hv.Points(
#     data=pd.DataFrame(gdf_traces), 
#     kdims=['trend_2011', 'trend_2016'], 
#     vdims=['Site']).groupby('Site')
# (
#     one_to_one.opts(color='black') 
#     * scatt_yr.opts(
#         xlim=(-0.035,0.01), ylim=(-0.035,0.01), 
#         xlabel='2011 trend (%/yr)', 
#         ylabel='2016 flight (%/yr)'))

# # %%
# sns.kdeplot(gdf_traces['trend_res'])

# print(
#     f"The mean error (in %/yr) for trend values is "
#     f"{100*((gdf_traces['t2011_ub']-gdf_traces['t2011_lb'])/2).mean():.2f}% "
#     f"for the 2011 manual estimates and "
#     f"{100*((gdf_traces['t2016_ub']-gdf_traces['t2016_lb'])/2).mean():.2f}% "
#     f"for the 2016 manual estimates")
# # %%

# plt_trend = gv.Points(
#     data=gdf_traces, 
#     vdims=['trend_res', 'trend_2011', 'trend_2016'], 
#     crs=ANT_proj).opts(
#         projection=ANT_proj, color='trend_res', 
#         cmap='coolwarm', symmetric=True, colorbar=True, tools=['hover'], 
#         width=600, height=400)
# plt_trend
