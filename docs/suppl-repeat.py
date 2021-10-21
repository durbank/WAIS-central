# Script for processing and analyzing additional repeat flightline data

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
from shapely.geometry import Point
hv.extension('bokeh', 'matplotlib')
gv.extension('bokeh', 'matplotlib')
import panel as pn
import seaborn as sns
import xarray as xr
from xrspatial import hillshade

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)

# Set project root directory
ROOT_DIR = Path(__file__).parents[1]

# Import custom project functions
import sys
SRC_DIR = ROOT_DIR.joinpath('src')
sys.path.append(str(SRC_DIR))
from my_mods import paipr, rema, viz, stats
from my_mods import spat_ops as so

# %%

def grsl_clean(
    data_dir, start_yr=None, end_yr=None, 
    yr_clip=True, rm_deep=True, cut_QC=2):

    # Import PAIPR results
    data_raw = paipr.import_PAIPR(data_dir)
    data_raw.query('QC_flag < @cut_QC', inplace=True)
    # data_0 = data_raw.query(
    #     'Year > QC_yr').sort_values(
    #     ['collect_time', 'Year']).reset_index(drop=True)
    data_0 = data_raw.sort_values(
        ['collect_time', 'Year']).reset_index(drop=True)
    data = paipr.format_PAIPR(
        data_0, start_yr=start_yr, end_yr=end_yr, 
        yr_clip=yr_clip, rm_deep=rm_deep).drop(
        'elev', axis=1)
    accum_df = data.pivot(
        index='Year', columns='trace_ID', values='accum')
    std_df = data.pivot(
        index='Year', columns='trace_ID', values='std')

    # Create gdf of mean results for each trace and 
    # transform to Antarctic Polar Stereographic
    gdf = paipr.long2gdf(data)
    gdf.to_crs(epsg=3031, inplace=True)

    return accum_df, std_df, gdf

# %% Additional central WAIS data

aPIG2009_ALL, stdPIG2009_ALL, gdf_PIG2009 = grsl_clean(
    ROOT_DIR.joinpath(
        'data/PAIPR-repeat/20091029/smb'), 
    start_yr=2004, end_yr=2008, yr_clip=False, rm_deep=True)
aPIG2016_ALL, stdPIG2016_ALL, gdf_PIG2016 = grsl_clean(
    ROOT_DIR.joinpath(
        'data/PAIPR-repeat/20161104/smb'), 
    start_yr=2004, end_yr=2008, yr_clip=False, rm_deep=True)

# Find nearest neighbors between 2011 and 2016 
# (within 500 m)
df_dist = so.nearest_neighbor(
    gdf_PIG2009, gdf_PIG2016, return_dist=True)
idx_paipr = df_dist['distance'] <= 500
dist_overlap = df_dist[idx_paipr]

# Create numpy arrays for relevant results
accumPIG_2009 = aPIG2009_ALL.iloc[
    :,dist_overlap.index]
stdPIG_2009 = stdPIG2009_ALL.iloc[
    :,dist_overlap.index]
accumPIG_2016 = aPIG2016_ALL.iloc[
    :,dist_overlap['trace_ID']]
stdPIG_2016 = stdPIG2016_ALL.iloc[
    :,dist_overlap['trace_ID']]

# Create new gdf of subsetted results
gdf_PAIPR_PIG = gpd.GeoDataFrame(
    {'ID_2009': dist_overlap.index.values, 
    'ID_2016': dist_overlap['trace_ID'].values, 
    'trace_dist': dist_overlap['distance'].values,
    'QC_2009': gdf_PIG2009.loc[
        dist_overlap.index,'QC_med'].values,
    'QC_2016': dist_overlap['QC_med'].values, 
    'accum_2009': accumPIG_2009.mean(axis=0).values, 
    'accum_2016': accumPIG_2016.mean(axis=0).values, 
    'std_2009': stdPIG_2009.mean(axis=0).values,
    'std_2016': stdPIG_2016.mean(axis=0).values},
    geometry=dist_overlap.geometry.values)

# Calculate bulk accum mean and accum residual
gdf_PAIPR_PIG['accum_mu'] = gdf_PAIPR_PIG[
    ['accum_2009', 'accum_2016']].mean(axis=1)
gdf_PAIPR_PIG['accum_res'] = (
    (gdf_PAIPR_PIG.accum_2016 - gdf_PAIPR_PIG.accum_2009) 
    / gdf_PAIPR_PIG.accum_mu)

# %%

gdf_PAIPR_PIG['accum_res'].plot(kind='kde')
# %%

plt_res = gv.Points(
    data=gdf_PAIPR_PIG, crs=ANT_proj, 
    vdims=['accum_res']).opts(
        projection=ANT_proj, color='accum_res', size=12,
        bgcolor='silver', 
        colorbar=True, 
        # cmap='seismic_r', 
        cmap='BrBG',
        symmetric=True, tools=['hover'], 
        width=750, height=750)
plt_res
  
# %%

# Create dataframes for scatter plots
PAIPR_df = pd.DataFrame(
    {'tmp_ID': np.tile(
        np.arange(0,accumPIG_2009.shape[1]), 
        accumPIG_2009.shape[0]), 
    'Year': np.reshape(
        np.repeat(accumPIG_2009.index, accumPIG_2009.shape[1]), 
        accumPIG_2009.size), 
    'accum_2009': 
        np.reshape(
            accumPIG_2009.values, accumPIG_2009.size), 
    'std_2009': np.reshape(
        stdPIG_2009.values, stdPIG_2009.size), 
    'accum_2016': np.reshape(
        accumPIG_2016.values, accumPIG_2016.size), 
    'std_2016': np.reshape(
        stdPIG_2016.values, stdPIG_2016.size)})

# Add residuals to dataframe
PAIPR_df['res_accum'] = PAIPR_df['accum_2016']-PAIPR_df['accum_2009']
PAIPR_df['res_perc'] = (
    100*(PAIPR_df['res_accum'])
    /(PAIPR_df[['accum_2016','accum_2009']]).mean(axis=1))

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))

scatt_yr = hv.Points(
    data=PAIPR_df, 
    kdims=['accum_2009', 'accum_2016'], 
    vdims=['Year'])
    
paipr_1to1_plt = one_to_one.opts(color='black')*scatt_yr.opts(
    xlim=(100,750), ylim=(100,750), 
    xlabel='2009 PAIPR (mm/yr)', 
    ylabel='2016 PAIPR (mm/yr)', 
    color='Year', cmap='plasma', colorbar=True, 
    width=700, height=700, fontscale=1.75)

# %%

PAIPR_df['res_perc'].plot(kind='kde')
# %%
