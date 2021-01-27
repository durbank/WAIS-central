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

# %%[markdown]
# ## 2011 PAIPR-manual comparisons
# 
# %% Import 20111109 results

# Import and format PAIPR results
dir1 = ROOT_DIR.joinpath('data/PAIPR-outputs/20111109/')
data_raw = import_PAIPR(dir1)
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)
data_2011 = format_PAIPR(
    data_0, start_yr=1985, end_yr=2010).drop(
    'elev', axis=1)
a2011_ALL = data_2011.pivot(
    index='Year', columns='trace_ID', values='accum')
std2011_ALL = data_2011.pivot(
    index='Year', columns='trace_ID', values='std')

# Import and format manual results
dir_0 = ROOT_DIR.joinpath('data/smb_manual/20111109/')
data_0 = import_PAIPR(dir_0)
man_2011 = format_PAIPR(
    data_0, start_yr=1985, end_yr=2010).drop(
    'elev', axis=1)
man2011_ALL = man_2011.pivot(
    index='Year', columns='trace_ID', values='accum')
manSTD_2011_ALL = man_2011.pivot(
    index='Year', columns='trace_ID', values='std')

# Create gdf of mean results for each trace and 
# transform to Antarctic Polar Stereographic
gdf_2011 = long2gdf(data_2011)
gdf_2011.to_crs(epsg=3031, inplace=True)
gdf_man2011 = long2gdf(man_2011)
gdf_man2011.to_crs(epsg=3031, inplace=True)

# %%
df_dist = nearest_neighbor(
    gdf_man2011, gdf_2011, return_dist=True)
idx_2011 = df_dist['distance'] <= 500
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
res_2011 = Maccum_2011.values - man2011_accum.values
accum_bar = np.mean(
    [Maccum_2011.mean(axis=0).values, 
    man2011_accum.mean(axis=0).values], 
    axis=0)
res2011_perc = res_2011 / accum_bar

fig, ax = plt.subplots()
ax = sns.kdeplot(
    res_2011.reshape(res_2011.size), color='blue')
fig.show()

print(
    f"The mean bias between manually-traced and "
    f"PAIPR-derived accumulation for 2011 is "
    f"{res_2011.mean():.2f} mm/yr ("
    f"{res2011_perc.mean()*100:.2f}% of the mean accumulation) "
    f"with a RMSE of {res2011_perc.std()*100:.2f}%"
)
print(
    f"The mean annual accumulation for 2011 results are "
    f"{Maccum_2011.values.mean():.0f} mm/yr for PAIPR " 
    f"and {man2011_accum.values.mean():.0f} mm\yr "
    f"for manual results."
)
print(
    f"The mean standard deviations of the annual "
    f"accumulation estimates are {(man2011_std/man2011_accum).values.mean()*100:.2f}% " 
    f"for 2011 manual layers and "
    f"{(Mstd_2011/Maccum_2011).values.mean()*100:.2f}% "
    f"for 2011 PAIPR layers."
)

#%%

import statsmodels.api as sm

a2011_1D = Maccum_2011.values.reshape(Maccum_2011.size)
q_M11 = np.quantile(
    (a2011_1D-a2011_1D.mean())/a2011_1D.std(), 
    np.arange(0,1,0.01))
man_1D = man2011_accum.values.reshape(man2011_accum.size)
q_man = np.quantile(
    (man_1D-man_1D.mean())/man_1D.std(), 
    np.arange(0,1,0.01))

plt.scatter(q_M11, q_man, color='blue')
plt.plot(
    [q_man.min(), q_man.max()], [q_man.min(), q_man.max()],
    color='red')


# %%

def plot_TScomp(ts_df1, ts_df2, gdf_combo, labels, colors=['blue', 'red']):
    """
    Stuff.
    """
    site_list = np.unique(gdf_combo['Site']).tolist()
    if "Null" in site_list:
        site_list.remove("Null")

    for site in site_list:

        idx = np.flatnonzero(gdf_combo['Site']==site)
        df1 = ts_df1.iloc[:,idx]
        df2 = ts_df2.iloc[:,idx]
        fig, ax = plt.subplots()
    
        df1.mean(axis=1).plot(ax=ax, color=colors[0], 
            linewidth=2, label=labels[0])
        (df1.mean(axis=1)+df1.std(axis=1)).plot(
            ax=ax, color=colors[0], linestyle='--', 
            label='__nolegend__')
        (df1.mean(axis=1)-df1.std(axis=1)).plot(
            ax=ax, color=colors[0], linestyle='--', 
            label='__nolegend__')

        df2.mean(axis=1).plot(
            ax=ax, color=colors[1], linewidth=2, 
            label=labels[1])
        (df2.mean(axis=1)+df2.std(axis=1)).plot(
            ax=ax, color=colors[1], linestyle='--', 
            label='__nolegend__')
        (df2.mean(axis=1)-df2.std(axis=1)).plot(
            ax=ax, color=colors[1], linestyle='--', 
            label='__nolegend__')
        ax.legend()
        fig.suptitle(site+' time series')
        fig.show()

# Assign flight chunk class based on location
chunk_centers = gpd.GeoDataFrame(
    ['PIG', 'SEAT2010-6', 'SEAT2010-5', 
    'SEAT2010-4', 'high-accum', 'mid-accum'],
    geometry=gpd.points_from_xy(
        [-1.297E6, -1.024E6, -1.093E6, 
            -1.159E6, -1.263E6, -1.177E6], 
        [-1.409E5, -4.639E5, -4.639E5, 
            -4.640E5, -4.868E5, -2.898E5]), 
    crs="EPSG:3031")

# %%

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

mid_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(
            chunk_centers.geometry.x[5], 
            gdf_traces2011.shape[0]), 
        np.repeat(
            chunk_centers.geometry.y[5], 
            gdf_traces2011.shape[0])), 
        crs="EPSG:3031")
    .distance(
        gdf_traces2011.reset_index())) <= 30000).values

gdf_traces2011['Site'] = np.repeat(
    'Null', gdf_traces2011.shape[0])
gdf_traces2011['Site'][PIG_idx] = 'PIG'
gdf_traces2011['Site'][SEAT6_idx] = 'SEAT2010-6'
gdf_traces2011['Site'][SEAT5_idx] = 'SEAT2010-5'
gdf_traces2011['Site'][SEAT4_idx] = 'SEAT2010-4'
gdf_traces2011['Site'][high_idx] = 'high-accum'
gdf_traces2011['Site'][mid_idx] = 'mid-accum'

plot_TScomp(
    Maccum_2011, man2011_accum, gdf_traces2011, 
    labels=['2011 PAIPR', '2011 manual'])

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

# %%[markdown]
# ## 2016 PAIPR-manual comparisons
# 
# %% Import 20161109 results

# Import and format PAIPR results
dir1 = ROOT_DIR.joinpath('data/PAIPR-outputs/20161109/')
data_raw = import_PAIPR(dir1)
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)
data_2016 = format_PAIPR(
    data_0, start_yr=1990, end_yr=2015).drop(
    'elev', axis=1)
a2016_ALL = data_2016.pivot(
    index='Year', columns='trace_ID', values='accum')
std2016_ALL = data_2016.pivot(
    index='Year', columns='trace_ID', values='std')

# Import and format manual results
dir_0 = ROOT_DIR.joinpath('data/smb_manual/20161109/')
data_0 = import_PAIPR(dir_0)
man_2016 = format_PAIPR(
    data_0, start_yr=1990, end_yr=2015).drop(
    'elev', axis=1)
man2016_ALL = man_2016.pivot(
    index='Year', columns='trace_ID', values='accum')
manSTD_2016_ALL = man_2016.pivot(
    index='Year', columns='trace_ID', values='std')

# Create gdf of mean results for each trace and 
# transform to Antarctic Polar Stereographic
gdf_2016 = long2gdf(data_2016)
gdf_2016.to_crs(epsg=3031, inplace=True)
gdf_man2016 = long2gdf(man_2016)
gdf_man2016.to_crs(epsg=3031, inplace=True)

# %%
df_dist2 = nearest_neighbor(
    gdf_man2016, gdf_2016, return_dist=True)
idx_2016 = df_dist2['distance'] <= 500
dist_overlap2 = df_dist2[idx_2016]
# dist_overlap2 = df_dist[idx_2011]

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
res_2016 = Maccum_2016.values - man2016_accum.values
accum_bar = np.mean(
    [Maccum_2016.mean(axis=0), 
    man2016_accum.mean(axis=0)], axis=0)
res2016_perc = res_2016 / accum_bar

sns.kdeplot(res2016_perc.reshape(res_2016.size))

print(
    f"The mean bias between manually-traced and "
    f"PAIPR-derived accumulation for 2016 is "
    f"{res_2016.mean():.2f} mm/yr ("
    f"{res2016_perc.mean()*100:.2f}% of mean accum) "
    f"with a RMSE of {res2016_perc.std()*100:.2f}%."
)
print(
    f"The mean annual accumulation for 2016 results are "
    f"{Maccum_2016.values.mean():.0f} mm/yr for PAIPR " 
    f"and {man2016_accum.values.mean():.0f} mm\yr "
    f"for manual results."
)
print(
    f"The mean standard deviations of the annual "
    f"accumulation estimates are "
    f"{(man2016_std/man2016_accum).values.mean()*100:.2f}% " 
    f"for 2016 manual layers and "
    f"{(Mstd_2016/Maccum_2016).values.mean()*100:.2f}% "
    f"for 2016 PAIPR layers."
)

# %%

SEAT4_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(
            chunk_centers.geometry.x[3], 
            gdf_traces2016.shape[0]), 
        np.repeat(
            chunk_centers.geometry.y[3], 
            gdf_traces2016.shape[0])), 
        crs="EPSG:3031")
    .distance(
        gdf_traces2016.reset_index())) <= 30000).values

high_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(
            chunk_centers.geometry.x[4], 
            gdf_traces2016.shape[0]), 
        np.repeat(
            chunk_centers.geometry.y[4], 
            gdf_traces2016.shape[0])), 
        crs="EPSG:3031")
    .distance(
        gdf_traces2016.reset_index())) <= 30000).values

mid_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(
            chunk_centers.geometry.x[5], 
            gdf_traces2016.shape[0]), 
        np.repeat(
            chunk_centers.geometry.y[5], 
            gdf_traces2016.shape[0])), 
        crs="EPSG:3031")
    .distance(
        gdf_traces2016.reset_index())) <= 30000).values

gdf_traces2016['Site'] = np.repeat(
    'Null', gdf_traces2016.shape[0])
gdf_traces2016['Site'][SEAT4_idx] = 'SEAT2010-4'
gdf_traces2016['Site'][high_idx] = 'high-accum'
gdf_traces2016['Site'][mid_idx] = 'mid-accum'

plot_TScomp(
    Maccum_2016, man2016_accum, gdf_traces2016, 
    labels=['2016 PAIPR', '2016 manual'])


# %%[markdown]
# ## Manual repeatability tests
# 
# %% Import manual results

# Get list of 2011 manual files
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

# %% Subset manual results to overlapping sections
df_dist = nearest_neighbor(
    gdf_man2016, gdf_man2011, return_dist=True)
idx_2016 = df_dist['distance'] <= 500
dist_overlap = df_dist[idx_2016]

# Create numpy arrays for relevant results
accum_man2011 = man2011_ALL.iloc[
    :,dist_overlap['trace_ID']]
std_man2011 = manSTD_2011_ALL.iloc[
    :,dist_overlap['trace_ID']]
accum_man2016 = man2016_ALL.iloc[
    :,dist_overlap.index]
std_man2016 = manSTD_2011_ALL.iloc[
    :,dist_overlap.index]

# Create new gdf of subsetted results
gdf_traces = gpd.GeoDataFrame(
    {'ID_2011': dist_overlap['trace_ID'].values, 
    'ID_2016': dist_overlap.index.values, 
    'accum_man2011': accum_man2011.mean(axis=0).values, 
    'accum_man2016': accum_man2016.mean(axis=0).values},
    geometry=dist_overlap.geometry.values)

# %%

# Calculate residuals (as % bias of mean accumulation)
man_res = accum_man2016.values - accum_man2011.values
accum_bar = np.mean(
    [accum_man2011.mean(axis=0).values, 
    accum_man2016.mean(axis=0).values], axis=0)
man_res_perc = man_res / accum_bar

sns.kdeplot(man_res_perc.reshape(man_res.size))

print(
    f"The mean bias in manually-derived annual accumulation 2016 vs 2011 flights is "
    f"{man_res.mean():.1f} mm/yr ("
    f"{man_res_perc.mean()*100:.2f}% of mean accum) "
    f"with a RMSE of {man_res_perc.std()*100:.2f}%."
)
print(
    f"The mean annual accumulation for manual results are "
    f"{accum_man2011.values.mean():.0f} mm/yr for 2011 " 
    f"and {accum_man2016.values.mean():.0f} mm/yr for 2016"
)
print(
    f"The mean standard deviations of the annual accumulation estimates are {(std_man2011/accum_man2011).values.mean()*100:.2f}% for 2011 and {(std_man2016/accum_man2016).values.mean()*100:.2f}% for 2016.")

# %%

# Assign flight chunk class based on location
chunk_centers = gpd.GeoDataFrame(
    ['PIG', 'SEAT2010-6', 'SEAT2010-5', 
    'SEAT2010-4', 'high-accum', 'mid-accum'],
    geometry=gpd.points_from_xy(
        [-1.297E6, -1.024E6, -1.093E6, 
            -1.159E6, -1.263E6, -1.177E6], 
        [-1.409E5, -4.639E5, -4.639E5, 
            -4.640E5, -4.868E5, -2.898E5]), 
    crs="EPSG:3031")

SEAT4_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(
            chunk_centers.geometry.x[3], 
            gdf_traces.shape[0]), 
        np.repeat(
            chunk_centers.geometry.y[3], 
        gdf_traces.shape[0])), 
        crs="EPSG:3031")
    .distance(gdf_traces.reset_index())) <= 30000).values
        
high_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(
            chunk_centers.geometry.x[4], 
        gdf_traces.shape[0]), 
        np.repeat(
            chunk_centers.geometry.y[4], 
            gdf_traces.shape[0])), 
        crs="EPSG:3031")
    .distance(gdf_traces.reset_index())) <= 30000).values

mid_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(
            chunk_centers.geometry.x[5], 
            gdf_traces.shape[0]), 
        np.repeat(
            chunk_centers.geometry.y[5], 
            gdf_traces.shape[0])), 
        crs="EPSG:3031")
    .distance(gdf_traces.reset_index())) <= 30000).values

gdf_traces['Site'] = np.repeat('Null', gdf_traces.shape[0])
gdf_traces['Site'][SEAT4_idx] = 'SEAT2010-4'
gdf_traces['Site'][high_idx] = 'high-accum'
gdf_traces['Site'][mid_idx] = 'mid-accum'

plot_TScomp(
    accum_man2011, accum_man2016, gdf_traces, 
    labels=['2011 manual', '2016 manual'])

# %%

# Create dataframes for scatter plots
accum_df = pd.DataFrame(
    {'Trace': np.tile(
        np.arange(0,accum_man2011.shape[1]), 
        accum_man2011.shape[0]), 
    'Site': np.tile(
        gdf_traces['Site'], accum_man2011.shape[0]), 
    'Year': np.reshape(
        np.repeat(man2011_ALL.index, accum_man2011.shape[1]), 
        accum_man2011.size), 
    'accum_2011': 
        np.reshape(
            accum_man2011.values, accum_man2011.size), 
    'std_2011': np.reshape(
        std_man2011.values, std_man2011.size), 
    'accum_2016': np.reshape(
        accum_man2016.values, accum_man2016.size), 
    'std_2016': np.reshape(
        std_man2016.values, std_man2016.size)})

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=accum_df, 
    kdims=['accum_2011', 'accum_2016'], 
    vdims=['Year']).groupby('Year')
(
    one_to_one.opts(color='black') 
    * scatt_yr.opts(
        xlim=(100,750), ylim=(100,750), 
        xlabel='2011 flight (mm/yr)', 
        ylabel='2016 flight (mm/yr)'))

# %%

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=accum_df, 
    kdims=['accum_2011', 'accum_2016'], 
    vdims=['Site']).groupby('Site')
(
    one_to_one.opts(color='black') 
    * scatt_yr.opts(
        xlim=(100,750), ylim=(100,750), 
        xlabel='2011 flight (mm/yr)', 
        ylabel='2016 flight (mm/yr)'))




# %%[markdown]
# ## PAIPR repeatability tests
# 
# %% 

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

# %%

df_dist = nearest_neighbor(
    gdf_2011, gdf_2016, return_dist=True)
idx_paipr = df_dist['distance'] <= 500
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
res_PAIPR = accum_2016.values - accum_2011.values 
accum_bar = np.mean(
    [accum_2016.mean(axis=0).values, 
    accum_2011.mean(axis=0).values], axis=0)
resPAIPR_perc = res_PAIPR / accum_bar

sns.kdeplot(resPAIPR_perc.reshape(res_PAIPR.size))

print(
    f"The mean bias between PAIPR-derived results "
    f"between 2016 and 2011 flights is "
    f"{res_PAIPR.mean():.1f} mm/yr ("
    f"{resPAIPR_perc.mean()*100:.2f}% of mean accum) "
    f"with a RMSE of {resPAIPR_perc.std()*100:.2f}%."
)
print(
    f"The mean annual accumulation for PAIPR results are "
    f"{accum_2011.values.mean():.0f} mm/yr for 2011 " 
    f"and {accum_2016.values.mean():.0f} mm/yr for 2016"
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

import random as rnd

# Diagnostic plots
plt_idxs = rnd.sample(range(accum_2011.shape[1]), 6)
for i, idx in enumerate(plt_idxs):
    fig, ax = plt.subplots()
    accum_2011.iloc[:,idx].plot(
        ax=ax, color='blue', linewidth=2, 
        label='2011 flight')
    (accum_2011.iloc[:,idx]+std_2011.iloc[:,idx]).plot(
        ax=ax, color='blue', linestyle='--', 
        label='__nolegend__')
    (accum_2011.iloc[:,idx]-std_2011.iloc[:,idx]).plot(
        ax=ax, color='blue', linestyle='--', 
        label='__nolegend__')
    accum_2016.iloc[:,idx].plot(
        ax=ax, color='red', linewidth=2, 
        label='2016 flight')
    (accum_2016.iloc[:,idx]+std_2016.iloc[:,idx]).plot(
        ax=ax, color='red', linestyle='--', 
        label='__nolegend__')
    (accum_2016.iloc[:,idx]-std_2016.iloc[:,idx]).plot(
        ax=ax, color='red', linestyle='--', 
        label='__nolegend__')
    ax.legend()
    fig.suptitle('PAIPR time series Example '+(str(i+1)))
    fig.show()

# %%

# Create dataframes for scatter plots
accum_df = pd.DataFrame(
    {'tmp_ID': np.tile(
        np.arange(0,accum_2011.shape[1]), 
        accum_2011.shape[0]), 
    'Year': np.reshape(
        np.repeat(accum_2011.index, accum_2011.shape[1]), 
        accum_2011.size), 
    'accum_2011': 
        np.reshape(
            accum_2011.values, accum_2011.size), 
    'std_2011': np.reshape(
        std_2011.values, std_2011.size), 
    'accum_2016': np.reshape(
        accum_2016.values, accum_2016.size), 
    'std_2016': np.reshape(
        std_2016.values, std_2016.size)})

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=accum_df, 
    kdims=['accum_2011', 'accum_2016'], 
    vdims=['Year']).groupby('Year')
(
    one_to_one.opts(color='black') 
    * scatt_yr.opts(
        xlim=(100,750), ylim=(100,750), 
        xlabel='2011 flight (mm/yr)', 
        ylabel='2016 flight (mm/yr)'))

# %%

# gdf_PAIPR['trend_2011'] = trend_2011 / accum_bar
# gdf_PAIPR['t2011_lb'] = lb_2011 / accum_bar
# gdf_PAIPR['t2011_ub'] = ub_2011 / accum_bar
# gdf_PAIPR['trend_2016'] = trend_2016 / accum_bar
# gdf_PAIPR['t2016_lb'] = lb_2016 / accum_bar
# gdf_PAIPR['t2016_ub'] = ub_2016 / accum_bar
# gdf_PAIPR['trend_res'] = trend_2016 - trend_2011

# plt.plot([-0.03,0.04], [-0.03,0.04], color='black')
# plt.scatter(gdf_PAIPR['trend_2011'], gdf_PAIPR['trend_2016'])
# plt.show()

# plt.plot([-20,5], [-20,5], color='black')
# plt.scatter(
#     gdf_PAIPR['trend_2011'] * accum_bar, 
#     gdf_PAIPR['trend_2016'] * accum_bar)
# plt.show()

# %%
# sns.kdeplot(gdf_PAIPR['trend_res'])

# print(
#     f"The mean trend bias between 2016 and 2011 "
#     f"PAIPR-derived estimates is "
#     f"{gdf_PAIPR['trend_res'].mean():.2f} mm/yr "
#     f"(~{100*gdf_PAIPR['trend_res'].mean()/gdf_PAIPR['accum_2011'].mean():.2f}% of the mean accum)")

# %%

# plt_trend = gv.Points(
#     data=gdf_PAIPR, 
#     vdims=['trend_res', 'trend_2011', 'trend_2016'], 
#     crs=ANT_proj).opts(
#         projection=ANT_proj, color='trend_res', 
#         cmap='coolwarm', symmetric=True, colorbar=True, bgcolor='silver', 
#         tools=['hover'], width=700, height=700)
# plt_trend
