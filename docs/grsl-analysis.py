# Script for generating results and figures for the Geoscience and Remote Sensing Letters article submission

# %% Set environment

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
hv.extension('bokeh', 'matplotlib')
gv.extension('bokeh', 'matplotlib')
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
# ## Data and Study Site
# 
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

# %% Determine overlap between datasets

# Find nearest neighbors between 2011 and 2016 
# (within 500 m)
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

# Create new gdf of subsetted results
gdf_PAIPR = gpd.GeoDataFrame(
    {'ID_2011': dist_overlap.index.values, 
    'ID_2016': dist_overlap['trace_ID'].values, 
    'accum_2011': 
        accum_2011.mean(axis=0).values, 
    'accum_2016': 
        accum_2016.mean(axis=0).values},
    geometry=dist_overlap.geometry.values)

# Calculate bulk accum mean and accum residual
gdf_PAIPR['accum_mu'] = gdf_PAIPR[
    ['accum_2011', 'accum_2016']].mean(axis=1)
gdf_PAIPR['accum_res'] = (
    (gdf_PAIPR.accum_2016 - gdf_PAIPR.accum_2011) 
    / gdf_PAIPR.accum_mu)

# Assign flight chunk label based on location
chunk_centers = gpd.GeoDataFrame({
    'Site': ['A', 'B', 'C', 'D', 'E', 'F'],
    'Name': ['A', 'B', 'C', 'D', 'E', 'F']},
    geometry=gpd.points_from_xy(
        [-1.177E6, -1.115E6, -1.024E6, -1.093E6, 
            -1.159E6, -1.263E6], 
        [-2.898E5, -3.681E5, -4.639E5, -4.639E5, 
            -4.640E5, -4.668E5]), 
    crs="EPSG:3031")

# # Assign flight chunk label based on location
# chunk_centers = gpd.GeoDataFrame({
#     'Site': ['PIG', 'A', 'B', 'C', 'D', 'E', 'F'],
#     'Name': ['PIG', 'A', 'B', 'C', 'D', 'E', 'F']},
#     geometry=gpd.points_from_xy(
#         [-1.297E6, -1.177E6, -1.115E6, -1.024E6, 
#             -1.093E6, -1.159E6, -1.263E6], 
#         [-1.409E5, -2.898E5, -3.681E5, -4.639E5, 
#             -4.639E5, -4.640E5, -4.668E5]), 
#     crs="EPSG:3031")

#%% Data location map


plt_accum2011 = gv.Points(
    gdf_2011, crs=ANT_proj, vdims=['accum']).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, size=1, 
        bgcolor='silver', tools=['hover'], 
        width=700, height=700)
plt_accum2016 = gv.Points(
    gdf_2016, crs=ANT_proj, vdims=['accum']).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, size=1,
        bgcolor='silver', tools=['hover'], 
        width=700, height=700)

plt_loc2011 = gv.Points(
    gdf_2011, crs=ANT_proj, vdims=['accum','std']).opts(
        projection=ANT_proj, color='blue', alpha=0.9, 
        size=5, bgcolor='silver', tools=['hover'], 
        width=700, height=700)
plt_loc2016 = gv.Points(
    gdf_2016, crs=ANT_proj, vdims=['accum','std']).opts(
        projection=ANT_proj, color='red', alpha=0.9, 
        size=5, bgcolor='silver', tools=['hover'], 
        width=700, height=700)
plt_locCOMB = gv.Points(
    gdf_PAIPR, crs=ANT_proj, 
    vdims=['accum_2011','accum_2016']).opts(
        projection=ANT_proj, color='mistyrose', 
        size=10, bgcolor='silver', tools=['hover'], 
        width=700, height=700)



plt_manPTS = gv.Points(
    chunk_centers, crs=ANT_proj, 
    vdims='Site').opts(
        projection=ANT_proj, color='black', 
        size=15, marker='square')

plt_labels = hv.Labels(
    {'x': chunk_centers.geometry.x.values, 
    'y': chunk_centers.geometry.y.values, 
    'text': chunk_centers.Site.values}, 
    ['x','y'], 'text').opts(
        yoffset=20000, text_color='black', 
        text_font_size='18pt')

# (
#     plt_locCOMB * plt_manPTS 
#     * (plt_accum2011 * plt_accum2016) * plt_labels
# )

# %% Experiments with matplotlib backend

# plt_accum2011 = gv.Points(
#     gdf_2011, crs=ANT_proj, vdims=['accum']).opts(
#         projection=ANT_proj, color='accum', 
#         cmap='viridis', colorbar=True, s=3)

# fig = hv.render(plt_accum2011, backend='matplotlib')
# fig.set_size_inches((10,10))


# %% PAIPR residuals

# Calculate residuals (as % bias of mean accumulation)
res_PAIPR = pd.DataFrame(
    accum_2016.values - accum_2011.values, 
    index=accum_2011.index) 
accum_bar = np.mean(
    [accum_2016.mean(axis=0).values, 
    accum_2011.mean(axis=0).values], axis=0)
resPAIPR_perc = 100*(res_PAIPR / accum_bar)

# %% Spatial distribution in PAIPR residuals

gdf_PAIPR['accum_mu'] = gdf_PAIPR[
    ['accum_2011', 'accum_2016']].mean(axis=1)
gdf_PAIPR['accum_res'] = 100*(
    (gdf_PAIPR.accum_2016 - gdf_PAIPR.accum_2011) 
    / gdf_PAIPR.accum_mu)

res_min = np.quantile(gdf_PAIPR.accum_res, 0.01)
res_max = np.quantile(gdf_PAIPR.accum_res, 0.99)

plt_accum = gv.Points(
    data=gdf_PAIPR, crs=ANT_proj, vdims=['accum_mu']).opts(
        projection=ANT_proj, color='accum_mu', 
        bgcolor='silver', colorbar=True, cmap='viridis', 
        tools=['hover'], width=700, height=700)

plt_res = gv.Points(
    data=gdf_PAIPR, crs=ANT_proj, vdims=['accum_res']).opts(
        projection=ANT_proj, color='accum_res', size=2,
        bgcolor='silver', colorbar=True, cmap='coolwarm_r', 
        symmetric=True, tools=['hover'], 
        width=600, height=600, fontsize=1.75)


plt_res = plt_res.redim.range(accum_res=(res_min,res_max))
# plt_res

# %%

# Generate indices corresponding to desired sites
gdf_PAIPR['Site'] = np.repeat(
    'Null', gdf_PAIPR.shape[0])
for label in chunk_centers['Site']:

    geom = chunk_centers.query('Site == @label').geometry
    idx = (gpd.GeoSeries(
        data=gpd.points_from_xy(
            np.repeat(geom.x, gdf_PAIPR.shape[0]), 
            np.repeat(geom.y, gdf_PAIPR.shape[0])), 
        crs=gdf_PAIPR.crs).distance(
            gdf_PAIPR.reset_index()) <= 30000).values
    
    gdf_PAIPR['Site'][idx] = label

# Create dataframes for scatter plots
PAIPR_df = pd.DataFrame(
    {'tmp_ID': np.tile(
        np.arange(0,accum_2011.shape[1]), 
        accum_2011.shape[0]), 
    'Site': np.tile(
        gdf_PAIPR['Site'], accum_2011.shape[0]), 
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
# scatt_yr = hv.Points(
#     data=PAIPR_df, 
#     kdims=['accum_2011', 'accum_2016'], 
#     vdims=['Year']).groupby('Year')
# (
#     one_to_one.opts(color='black') 
#     * scatt_yr.opts(
#         xlim=(100,750), ylim=(100,750), 
#         xlabel='2011 flight (mm/yr)', 
#         ylabel='2016 flight (mm/yr)'))

scatt_yr = hv.Points(
    data=PAIPR_df, 
    kdims=['accum_2011', 'accum_2016'], 
    vdims=['Year'])
    
paipr_1to1_plt = one_to_one.opts(color='black')*scatt_yr.opts(
    xlim=(100,750), ylim=(100,750), 
    xlabel='2011 PAIPR (mm/yr)', 
    ylabel='2016 PAIPR (mm/yr)', 
    color='Year', cmap='plasma', colorbar=True, 
    width=600, height=600, fontscale=1.75)

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=PAIPR_df, 
    kdims=['accum_2011', 'accum_2016'], 
    vdims=['Year', 'Site']).groupby('Site')
site_res_plt = one_to_one.opts(color='black')*scatt_yr.opts(
    xlim=(100,750), ylim=(100,750), 
    xlabel='2011 flight (mm/yr)', 
    ylabel='2016 flight (mm/yr)', 
    color='Year', cmap='plasma', colorbar=True, 
    width=600, height=600, fontscale=1.75)

paipr_1to1_plt_comb = paipr_1to1_plt + site_res_plt
# paipr_1to1_plt_comb

# %%

# import random as rnd

# # Diagnostic plots
# plt_idxs = rnd.sample(range(accum_2011.shape[1]), 6)
# for i, idx in enumerate(plt_idxs):
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
#     fig.suptitle('PAIPR time series Example '+(str(i+1)))
#     fig.show()

# %%[markdown]
# ## 2011 PAIPR-manual comparions
# 
# %%

# Import and format PAIPR results
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

# Import and format manual results
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
gdf_2011 = long2gdf(data_2011)
gdf_2011.to_crs(epsg=3031, inplace=True)
gdf_man2011 = long2gdf(man_2011)
gdf_man2011.to_crs(epsg=3031, inplace=True)

# %% Subset to overlapping data

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
res_2011 = pd.DataFrame(
    Maccum_2011.values - man2011_accum.values, 
    index=Maccum_2011.index)
accum_bar = np.mean(
    [Maccum_2011.mean(axis=0).values, 
    man2011_accum.mean(axis=0).values], 
    axis=0)
res2011_perc = 100*(res_2011 / accum_bar)

# %%

def plot_TScomp(
    ts_df1, ts_df2, gdf_combo, labels, 
    colors=['blue', 'red'], ts_err1=None, ts_err2=None):
    """This is a function to generate matplotlib objects that compare spatially overlapping accumulation time series.

    Args:
        ts_df1 (pandas.DataFrame): Dataframe containing time series for the first dataset.
        ts_df2 (pandas.DataFrame): Dataframe containing time series for the second dataset.
        gdf_combo (geopandas.geoDataFrame): Geodataframe with entries corresponding to the paired time series locations. Also contains a column 'Site' that groups the different time series according to their manual tracing location.
        labels (list of str): The labels used in the output plot to differentiate the time series dataframes.
        colors (list, optional): The colors to use when plotting the time series. Defaults to ['blue', 'red'].
        ts_err1 (pandas.DataFrame, optional): DataFrame containing time series errors corresponding to ts_df1. If "None" then the error is estimated from the standard deviations in annual results. Defaults to None.
        ts_err2 (pandas.DataFrame, optional): DataFrame containing time series errors corresponding to ts_df2. If "None" then the error is estimated from the standard deviations in annual results.. Defaults to None.

    Returns:
        matplotlib.pyplot.figure: Generated figure comparing the two overlapping time series.
    """

    # Remove observations without an assigned site
    site_list = np.unique(gdf_combo['Site']).tolist()
    if "Null" in site_list:
        site_list.remove("Null")

    # Generate figure with a row for each site
    fig, axes = plt.subplots(
        ncols=1, nrows=len(site_list), 
        constrained_layout=True, 
        figsize=(6,24))

    for i, site in enumerate(site_list):
        
        # Subset results to specific site
        idx = np.flatnonzero(gdf_combo['Site']==site)
        df1 = ts_df1.iloc[:,idx]
        df2 = ts_df2.iloc[:,idx]

        # Check if errors for time series 1 are provided
        if ts_err1 is not None:

            df_err1 = ts_err1.iloc[:,idx]

            # Plot ts1 and mean errors
            df1.mean(axis=1).plot(ax=axes[i], color=colors[0], 
                linewidth=2, label=labels[0])
            (df1.mean(axis=1)+df_err1.mean(axis=1)).plot(
                ax=axes[i], color=colors[0], linestyle='--', 
                label='__nolegend__')
            (df1.mean(axis=1)-df_err1.mean(axis=1)).plot(
                ax=axes[i], color=colors[0], linestyle='--', 
                label='__nolegend__')
        else:
            # If ts1 errors are not given, estimate as the 
            # standard deviation of annual estimates
            df1.mean(axis=1).plot(ax=axes[i], color=colors[0], 
                linewidth=2, label=labels[0])
            (df1.mean(axis=1)+df1.std(axis=1)).plot(
                ax=axes[i], color=colors[0], linestyle='--', 
                label='__nolegend__')
            (df1.mean(axis=1)-df1.std(axis=1)).plot(
                ax=axes[i], color=colors[0], linestyle='--', 
                label='__nolegend__')

        # Check if errors for time series 2 are provided
        if ts_err2 is not None:
            df_err2 = ts_err2.iloc[:,idx]

            # Plot ts2 and mean errors
            df2.mean(axis=1).plot(
                ax=axes[i], color=colors[1], linewidth=2, 
                label=labels[1])
            (df2.mean(axis=1)+df_err2.mean(axis=1)).plot(
                ax=axes[i], color=colors[1], linestyle='--', 
                label='__nolegend__')
            (df2.mean(axis=1)-df_err2.mean(axis=1)).plot(
                ax=axes[i], color=colors[1], linestyle='--', 
                label='__nolegend__')
        else:
            # If ts2 errors are not given, estimate as the 
            # standard deviation of annual estimates
            df2.mean(axis=1).plot(
                ax=axes[i], color=colors[1], linewidth=2, 
                label=labels[1])
            (df2.mean(axis=1)+df2.std(axis=1)).plot(
                ax=axes[i], color=colors[1], linestyle='--', 
                label='__nolegend__')
            (df2.mean(axis=1)-df2.std(axis=1)).plot(
                ax=axes[i], color=colors[1], linestyle='--', 
                label='__nolegend__')
        
        # Add legend and set title based on site name
        axes[i].legend()
        axes[i].set_title('Site '+site+' time series')

    return fig

# %% 2011 comparison plots

# Generate indices of corresponding to desiered sites
gdf_traces2011['Site'] = np.repeat(
    'Null', gdf_traces2011.shape[0])
for label in chunk_centers['Site']:

    geom = chunk_centers.query('Site == @label').geometry
    idx = (gpd.GeoSeries(
        data=gpd.points_from_xy(
            np.repeat(geom.x, gdf_traces2011.shape[0]), 
            np.repeat(geom.y, gdf_traces2011.shape[0])), 
        crs=gdf_traces2011.crs).distance(
            gdf_traces2011.reset_index()) <= 30000).values
    
    gdf_traces2011['Site'][idx] = label


tsfig_2011 = plot_TScomp(
    man2011_accum, Maccum_2011, gdf_traces2011, 
    labels=['2011 manual', '2011 PAIPR'], 
    ts_err1=man2011_std)

# %%

# Create dataframes for scatter plots
tmp_df1 = pd.DataFrame(
    {'tmp_ID': np.tile(
        np.arange(0,man2011_accum.shape[1]), 
        man2011_accum.shape[0]), 
    'Site': np.tile(
        gdf_traces2011['Site'], man2011_accum.shape[0]), 
    'Year': np.reshape(
        np.repeat(man2011_accum.index, man2011_accum.shape[1]), 
        man2011_accum.size), 
    'accum_man': 
        np.reshape(
            man2011_accum.values, man2011_accum.size), 
    'std_man': np.reshape(
        man2011_std.values, man2011_std.size), 
    'accum_paipr': np.reshape(
        Maccum_2011.values, Maccum_2011.size), 
    'std_paipr': np.reshape(
        Mstd_2011.values, Mstd_2011.size)})

tmp_df1['flight'] = 2011

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
    data_0, start_yr=1990, end_yr=2010).drop(
    'elev', axis=1)
a2016_ALL = data_2016.pivot(
    index='Year', columns='trace_ID', values='accum')
std2016_ALL = data_2016.pivot(
    index='Year', columns='trace_ID', values='std')

# Import and format manual results
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
gdf_2016 = long2gdf(data_2016)
gdf_2016.to_crs(epsg=3031, inplace=True)
gdf_man2016 = long2gdf(man_2016)
gdf_man2016.to_crs(epsg=3031, inplace=True)

# %%

df_dist2 = nearest_neighbor(
    gdf_man2016, gdf_2016, return_dist=True)
idx_2016 = df_dist2['distance'] <= 500
dist_overlap2 = df_dist2[idx_2016]

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
res_2016 = pd.DataFrame(
    Maccum_2016.values - man2016_accum.values, 
    index=Maccum_2016.index)
accum_bar = np.mean(
    [Maccum_2016.mean(axis=0), 
    man2016_accum.mean(axis=0)], axis=0)
res2016_perc = 100*(res_2016 / accum_bar)

# %% 2016 comparison plots

# Generate indices of corresponding to desiered sites
gdf_traces2016['Site'] = np.repeat(
    'Null', gdf_traces2016.shape[0])
for label in chunk_centers['Site']:

    geom = chunk_centers.query('Site == @label').geometry
    idx = (gpd.GeoSeries(
        data=gpd.points_from_xy(
            np.repeat(geom.x, gdf_traces2016.shape[0]), 
            np.repeat(geom.y, gdf_traces2016.shape[0])), 
        crs=gdf_traces2016.crs).distance(
            gdf_traces2016.reset_index()) <= 30000).values
    
    gdf_traces2016['Site'][idx] = label


tsfig_2016 = plot_TScomp(
    man2016_accum, Maccum_2016, gdf_traces2016, 
    labels=['2016 manual', '2016 PAIPR'], 
    ts_err1=man2016_std)

# %%

# Create dataframes for scatter plots
tmp_df2 = pd.DataFrame(
    {'tmp_ID': np.tile(
        (tmp_df1['tmp_ID'].max()+1) 
        + np.arange(0,man2016_accum.shape[1]), 
        man2016_accum.shape[0]), 
    'Site': np.tile(
        gdf_traces2016['Site'], man2016_accum.shape[0]), 
    'Year': np.reshape(
        np.repeat(man2016_accum.index, man2016_accum.shape[1]), 
        man2016_accum.size), 
    'accum_man': 
        np.reshape(
            man2016_accum.values, man2016_accum.size), 
    'std_man': np.reshape(
        man2016_std.values, man2016_std.size), 
    'accum_paipr': np.reshape(
        Maccum_2016.values, Maccum_2016.size), 
    'std_paipr': np.reshape(
        Mstd_2016.values, Mstd_2016.size)})

tmp_df2['flight'] = 2016

PAP_man_df = pd.concat([tmp_df1, tmp_df2], axis=0)

# %%

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))

scatt_yr = hv.Points(
    data=PAP_man_df, 
    kdims=['accum_man', 'accum_paipr'], 
    vdims=['Year'])
    
PM_1to1_plt = one_to_one.opts(color='black')*scatt_yr.opts(
    xlim=(100,750), ylim=(100,750), 
    xlabel='Manual accum (mm/yr)', 
    ylabel='PAIPR accum (mm/yr)', 
    color='Year', cmap='plasma', colorbar=True, 
    width=600, height=600, fontscale=1.75)

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=PAP_man_df, 
    kdims=['accum_man', 'accum_paipr'], 
    vdims=['Year', 'Site']).groupby('Site')
site_res_plt = one_to_one.opts(color='black')*scatt_yr.opts(
    xlim=(100,750), ylim=(100,750), 
    xlabel='Manual accum (mm/yr)', 
    ylabel='PAIPR accum (mm/yr)', 
    color='Year', cmap='plasma', colorbar=True, 
    width=600, height=600, fontscale=1.75)

PM_1to1_comb_plt = PM_1to1_plt + site_res_plt
# PM_1to1_comb_plt

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
std_man2016 = manSTD_2016_ALL.iloc[
    :,dist_overlap.index]

# Create new gdf of subsetted results
gdf_MANtraces = gpd.GeoDataFrame(
    {'ID_2011': dist_overlap['trace_ID'].values, 
    'ID_2016': dist_overlap.index.values, 
    'accum_man2011': accum_man2011.mean(axis=0).values, 
    'accum_man2016': accum_man2016.mean(axis=0).values},
    geometry=dist_overlap.geometry.values)

# %% Manual-manual comparison plots

# Generate indices of corresponding to desiered sites
gdf_MANtraces['Site'] = np.repeat(
    'Null', gdf_MANtraces.shape[0])
for label in chunk_centers['Site']:

    geom = chunk_centers.query('Site == @label').geometry
    idx = (gpd.GeoSeries(
        data=gpd.points_from_xy(
            np.repeat(geom.x, gdf_MANtraces.shape[0]), 
            np.repeat(geom.y, gdf_MANtraces.shape[0])), 
        crs=gdf_MANtraces.crs).distance(
            gdf_MANtraces.reset_index()) <= 30000).values
    
    gdf_MANtraces['Site'][idx] = label


tsfig_manual = plot_TScomp(
    accum_man2011, accum_man2016, gdf_MANtraces, 
    labels=['2011 manual', '2016 manual'])

# %% 

# Calculate residuals (as % bias of mean accumulation)
man_res = pd.DataFrame(
    accum_man2016.values-accum_man2011.values, 
    index=accum_man2011.index)
accum_bar = np.mean(
    [accum_man2011.mean(axis=0).values, 
    accum_man2016.mean(axis=0).values], axis=0)
man_res_perc = 100*(man_res / accum_bar)

# %% Repeat stats with SEAT10-4 removed

# Remove Site E (SEAT2010-4) from comparisons
idx_tmp2011 = gdf_MANtraces.query('Site != "E"')['ID_2011']
accum_tmp2011 = man2011_ALL.iloc[:,idx_tmp2011]
idx_tmp2016 = gdf_MANtraces.query('Site != "E"')['ID_2016']
accum_tmp2016 = man2016_ALL.iloc[:,idx_tmp2016]

# Calculate residuals (as % bias of mean accumulation)
res_tmp = pd.DataFrame(
    accum_tmp2016.values-accum_tmp2011.values, 
    index=accum_man2011.index)
bar_tmp = np.mean(
    [accum_tmp2011.mean(axis=0).values, 
    accum_tmp2016.mean(axis=0).values], axis=0)
res_perc_tmp = 100*(res_tmp / bar_tmp)

# %%

# Create dataframes for scatter plots
man_df = pd.DataFrame(
    {'Trace': np.tile(
        np.arange(0,accum_man2011.shape[1]), 
        accum_man2011.shape[0]), 
    'Site': np.tile(
        gdf_MANtraces['Site'], accum_man2011.shape[0]), 
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
    data=man_df, 
    kdims=['accum_2011', 'accum_2016'], 
    vdims=['Year'])

man_1to1_plt = one_to_one.opts(color='black')*scatt_yr.opts(
    xlim=(100,750), ylim=(100,750), 
    xlabel='2011 manual (mm/yr)', 
    ylabel='2016 manual (mm/yr)', 
    color='Year', cmap='plasma', colorbar=True, 
    width=600, height=600, fontscale=1.75)

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=man_df, 
    kdims=['accum_2011', 'accum_2016'], 
    vdims=['Year', 'Site']).groupby('Site')
site_res_plt = one_to_one.opts(color='black')*scatt_yr.opts(
    xlim=(100,750), ylim=(100,750), 
    xlabel='2011 Manual flight (mm/yr)', 
    ylabel='2016 Manual flight (mm/yr)', 
    color='Year', cmap='plasma', colorbar=True, 
    width=600, height=600, fontscale=1.75)

man_1to1_comb_plt = man_1to1_plt + site_res_plt
# man_1to1_comb_plt

# %%

tsfig_PAIPR = plot_TScomp(
    accum_2011, accum_2016, gdf_PAIPR, 
    labels=['2011 PAIPR', '2016 PAIPR'])

# %%[markdown]
# ## Final figures used in article
# 
# %% Data location map

data_map = (
    plt_locCOMB * plt_manPTS 
    * (plt_accum2011 * plt_accum2016) 
    * plt_labels.opts(text_font_size='32pt')
)

data_map.opts(fontscale=2.5, width=1200, height=1200)

# %% Density plots

plt.rcParams.update({'font.size': 22})
fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot()
sns.kdeplot(
    ax=ax1, 
    data=resPAIPR_perc.values.reshape(resPAIPR_perc.size), 
    label='2016-2011 PAIPR', linewidth=3)
ax2 = fig.add_subplot()
sns.kdeplot(
    ax=ax2, 
    data=res2011_perc.values.reshape(res2011_perc.size), 
    label='2011 PAIPR-manual', linewidth=3)   
ax3 = fig.add_subplot()
sns.kdeplot(
    ax=ax3, 
    data=res2016_perc.values.reshape(res2016_perc.size), 
    label='2016 PAIPR-manual', linewidth=3)
ax4 = fig.add_subplot()
sns.kdeplot(
    ax=ax4, 
    data=man_res_perc.values.reshape(man_res_perc.size), 
    label='2016-2011 manual', linewidth=3)
ax1.legend()
ax1.set_xlabel('Residual (% of mean annual accum)')

# %%

print(
    f"The mean bias between PAIPR-derived results "
    f"between 2016 and 2011 flights is "
    f"{res_PAIPR.values.mean():.1f} mm/yr ("
    f"{resPAIPR_perc.values.mean():.2f}% of mean accum) "
    f"with a RMSE of {resPAIPR_perc.values.std():.2f}%."
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

print(
    f"The mean bias between manually-traced and "
    f"PAIPR-derived accumulation for 2011 is "
    f"{res_2011.values.mean():.2f} mm/yr ("
    f"{res2011_perc.values.mean():.2f}% of the mean accumulation) "
    f"with a RMSE of {res2011_perc.values.std():.2f}%"
)
print(
    f"The mean annual accumulation for 2011 results are "
    f"{Maccum_2011.values.mean():.0f} mm/yr for PAIPR " 
    f"and {man2011_accum.values.mean():.0f} mm/yr "
    f"for manual results."
)
print(
    f"The mean standard deviations of the annual "
    f"accumulation estimates are {(man2011_std/man2011_accum).values.mean()*100:.2f}% " 
    f"for 2011 manual layers and "
    f"{(Mstd_2011/Maccum_2011).values.mean()*100:.2f}% "
    f"for 2011 PAIPR layers."
)

print(
    f"The mean bias between manually-traced and "
    f"PAIPR-derived accumulation for 2016 is "
    f"{res_2016.values.mean():.2f} mm/yr ("
    f"{res2016_perc.values.mean():.2f}% of mean accum) "
    f"with a RMSE of {res2016_perc.values.std():.2f}%."
)
print(
    f"The mean annual accumulation for 2016 results are "
    f"{Maccum_2016.values.mean():.0f} mm/yr for PAIPR " 
    f"and {man2016_accum.values.mean():.0f} mm/yr "
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

print(
    f"The mean bias in manually-derived annual accumulation 2016 vs 2011 flights is "
    f"{man_res.values.mean():.1f} mm/yr ("
    f"{man_res_perc.values.mean():.2f}% of mean accum) "
    f"with a RMSE of {man_res_perc.values.std():.2f}%."
)
print(
    f"The mean annual accumulation for manual results are "
    f"{accum_man2011.values.mean():.0f} mm/yr for 2011 " 
    f"and {accum_man2016.values.mean():.0f} mm/yr for 2016"
)
print(
    f"The mean standard deviations of the annual accumulation estimates are {(std_man2011/accum_man2011).values.mean()*100:.2f}% for 2011 and {(std_man2016/accum_man2016).values.mean()*100:.2f}% for 2016.")

print(
    f"The mean bias in manually-derived annual accumulation 2016 vs 2011 flights is "
    f"{res_tmp.values.mean():.1f} mm/yr ("
    f"{res_perc_tmp.values.mean():.2f}% of mean accum) "
    f"with a RMSE of {res_perc_tmp.values.std():.2f}%."
)
print(
    f"The mean annual accumulation for manual results are "
    f"{accum_tmp2011.values.mean():.0f} mm/yr for 2011 " 
    f"and {accum_tmp2016.values.mean():.0f} mm/yr for 2016"
)
print(
    f"The mean standard deviations of the annual accumulation estimates are {(std_man2011/accum_man2011).values.mean()*100:.2f}% for 2011 and {(std_man2016/accum_man2016).values.mean()*100:.2f}% for 2016.")

# %%

res_grid_plt = (
    plt_manPTS.opts(color='black')*
    plt_res.opts(size=5)
    * plt_labels.opts(
        text_color='black', text_font_size='22pt')).opts(
            fontscale=1.75)

results_plt = hv.Layout(
    paipr_1to1_plt+PM_1to1_plt+man_1to1_plt+res_grid_plt).cols(2)
results_plt

# %%

# tsfig_PAIPR.savefig(
#     fname=ROOT_DIR.joinpath(
#         'docs/Figures/oib-repeat/tsfig_PAIPR.svg'))
# tsfig_manual.savefig(
#     fname=ROOT_DIR.joinpath(
#         'docs/Figures/oib-repeat/tsfig_man.svg'))
# tsfig_2011.savefig(
#     fname=ROOT_DIR.joinpath(
#         'docs/Figures/oib-repeat/tsfig_2011.svg'))
# tsfig_2016.savefig(
#     fname=ROOT_DIR.joinpath(
#         'docs/Figures/oib-repeat/tsfig_2016.svg'))

# %%

# fig = plt.figure(constrained_layout=True)
# gs = fig.add_gridspec(6,4)

# fig_list = [tsfig_PAIPR, tsfig_manual, tsfig_2011, tsfig_2016]

# for i, fig_i in enumerate(fig_list):
#     axes = fig_i.get_axes()
#     for j, ax in enumerate(axes):
#         gs[j,i] = ax
#         # fig.add_subplot(gs[j,i])







# fig_list = [tsfig_PAIPR, tsfig_manual, tsfig_2011, tsfig_2016]

# fig, axes = plt.subplots(
#     ncols=len(fig_list), nrows=6, constrained_layout=True, 
#     figsize=(20, 30))

# %%

from bokeh.io import export_svgs

def export_svg(obj, filename):
    plot_state = hv.renderer('bokeh').get_plot(obj).state
    plot_state.output_backend = 'svg'
    export_svgs(plot_state, filename=filename)

# p = hv.render(data_map, backend='matplotlib')

# export_svg(data_map, ROOT_DIR.joinpath(
#     'docs/Figures/oib-repeat/data_map.svg'))

hv.save(data_map, ROOT_DIR.joinpath(
    'docs/Figures/oib-repeat/data_map.png'))
hv.save(results_plt, ROOT_DIR.joinpath(
    'docs/Figures/oib-repeat/results.png'))

# %%
