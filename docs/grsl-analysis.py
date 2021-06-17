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
from shapely.geometry import Point
# from bokeh.io import output_notebook
# output_notebook()
hv.extension('bokeh', 'matplotlib')
gv.extension('bokeh', 'matplotlib')
# hv.extension('matplotlib')
# gv.extension('matplotlib')
import panel as pn
import seaborn as sns
import xarray as xr
from xrspatial import hillshade
from scipy.spatial.distance import pdist

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)

# Set project root directory
ROOT_DIR = Path(__file__).parents[1]

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
dir1 = ROOT_DIR.joinpath('data/PAIPR-repeat/20111109/smb/')
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
dir2 = ROOT_DIR.joinpath('data/PAIPR-repeat/20161109/smb/')
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
    'trace_dist': dist_overlap['distance'].values,
    'QC_2011': gdf_2011.loc[dist_overlap.index,'QC_med'].values,
    'QC_2016': dist_overlap['QC_med'].values, 
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
# (sorted from high to low accumulation as 
# determined with 2011 manual results) 
chunk_centers = gpd.GeoDataFrame({
    'Site': ['A', 'B', 'C', 'D', 'E', 'F'],
    'Name': ['A', 'B', 'C', 'D', 'E', 'F']},
    geometry=gpd.points_from_xy(
        [-1.263E6, -1.159E6, -1.177E6, -1.115E6, 
            -1.093E6, -1.024E6], 
        [-4.668E5, -4.640E5, -2.898E5, -3.681E5, 
            -4.639E5, -4.639E5]), 
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

#%% Data map components

# Accumulation plots for both flights
plt_accum2011 = gv.Points(
    gdf_2011, crs=ANT_proj, vdims=['accum']).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, size=12, 
        bgcolor='silver', 
        tools=['hover'], 
        width=700, height=700)
plt_accum2016 = gv.Points(
    gdf_2016, crs=ANT_proj, vdims=['accum']).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, size=12,
        bgcolor='silver', 
        tools=['hover'], 
        width=700, height=700)

# Location plots for flights (and combined overlap)
plt_loc2011 = gv.Points(
    gdf_2011, crs=ANT_proj, vdims=['accum','std']).opts(
        projection=ANT_proj, color='blue', alpha=0.9, 
        size=5, 
        bgcolor='silver', 
        tools=['hover'], 
        width=700, height=700)
plt_loc2016 = gv.Points(
    gdf_2016, crs=ANT_proj, vdims=['accum','std']).opts(
        projection=ANT_proj, color='red', alpha=0.9, 
        size=5, 
        bgcolor='silver', 
        tools=['hover'], 
        width=700, height=700)
plt_locCOMB = gv.Points(
    gdf_PAIPR, crs=ANT_proj, 
    vdims=['accum_2011','accum_2016']).opts(
        projection=ANT_proj, color='cyan', 
        size=24, alpha=0.5,
        bgcolor='silver', 
        tools=['hover'], 
        width=700, height=700)


# Locations of manual tracing sites
plt_manPTS = gv.Points(
    chunk_centers, crs=ANT_proj, 
    vdims='Site').opts(
        projection=ANT_proj, color='black', 
        size=32, marker='square')

# Labels for manual tracing locations
plt_labels = hv.Labels(
    {'x': chunk_centers.geometry.x.values, 
    'y': chunk_centers.geometry.y.values, 
    'text': chunk_centers.Site.values}, 
    ['x','y'], 'text').opts(
        yoffset=20000, text_color='black', 
        text_font_size='28pt')

# Location of example echograms in Fig 2
echo_pt = gpd.GeoDataFrame(
    gpd.GeoSeries(Point(-113.250, -79.215), crs='epsg:4326'))
echo_pt = echo_pt.rename(
    columns={0:'geometry'}).set_geometry('geometry')
echo_pt.to_crs(epsg=3031, inplace=True)
plt_echo = gv.Points(echo_pt, crs=ANT_proj).opts(
    projection=ANT_proj, 
    color='white', marker='star', size=20)

# %%

# Define Antarctic DEM file
xr_DEM = xr.open_rasterio(
    ROOT_DIR.joinpath(
        'data/Antarctica_Cryosat2_1km_DEMv1.0.tif')).squeeze()

# Get bounds (with buffer) of data radar set
gdf_bounds = {
    'x_range': tuple(np.round(
        gdf_2016.total_bounds[0::2]+[-25000,25000])), 
    'y_range': tuple(np.round(
        gdf_2016.total_bounds[1::2]+[-25000,25000]))}

# Clip elevation data to radar bounds
xr_DEM = xr_DEM.sel(
    x=slice(gdf_bounds['x_range'][0], gdf_bounds['x_range'][1]), 
    y=slice(gdf_bounds['y_range'][1], gdf_bounds['y_range'][0]))

tpl_bnds = (
    gdf_bounds['x_range'][0], gdf_bounds['y_range'][0], 
    gdf_bounds['x_range'][1], gdf_bounds['y_range'][1])
elev_plt = hv.Image(xr_DEM.values, bounds=tpl_bnds).opts(
    cmap='dimgray', colorbar=False, width=700, height=700)

# Generate contour plot
cont_plt = hv.operation.contours(elev_plt, levels=15).opts(
    cmap='magma', show_legend=False, 
    colorbar=True, line_width=3)

# Generate elevation hillshade
xr_HS = hillshade(xr_DEM)
hill_plt = hv.Image(xr_HS.values, bounds=tpl_bnds).opts(
        alpha=0.25, cmap='gray', colorbar=False)

# %% Get ice velocities of trace locations

xr_vice = xr.open_dataset(ROOT_DIR.joinpath(
    'data/ice-velocities/antarctica_ice_velocity_450m_v2.nc'))
xr_clip = xr_vice.sel(
    x=slice(
        gdf_PAIPR.total_bounds[0], 
        gdf_PAIPR.total_bounds[2]), 
    y=slice(
        gdf_PAIPR.total_bounds[3], 
        gdf_PAIPR.total_bounds[1]))
xr_pts = extract_at_pts(xr_clip, gdf_PAIPR, return_dist=True)
vice_pts = gpd.GeoDataFrame(
    data=xr_pts[['VX', 'VY']], crs='epsg:3031', 
    geometry=xr_pts.geometry)
vice_pts['Vxy'] = np.sqrt(
    vice_pts['VX']**2 + vice_pts['VY']**2)

gv.Points(vice_pts, crs=ANT_proj, 
vdims=['Vxy', 'VX', 'VY']).opts(
    projection=ANT_proj, color='Vxy', cmap='viridis', 
    size=10, colorbar=True, logz=True,
    width=800, height=800, tools=['hover'])

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
        # bgcolor='silver', 
        colorbar=True, cmap='viridis', 
        tools=['hover'], width=700, height=700)

plt_res = gv.Points(
    data=gdf_PAIPR, crs=ANT_proj, vdims=['accum_res']).opts(
        projection=ANT_proj, color='accum_res', size=12,
        bgcolor='silver', 
        colorbar=True, 
        # cmap='seismic_r', 
        cmap='BrBG',
        symmetric=True, tools=['hover'], 
        width=600, height=600)


plt_res = plt_res.redim.range(accum_res=(res_min,res_max))
# plt_res

# %% Bias trend with QC rating

fig, ax = plt.subplots()
ax.scatter(x=abs(gdf_PAIPR['QC_2016']-gdf_PAIPR['QC_2011']), y=abs(gdf_PAIPR['accum_res']), color='red', alpha=0.25)
ax.hlines(y=0, xmin=0, xmax=0.08, color='black')

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
    
    gdf_PAIPR.loc[idx,'Site'] = label

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

# Add residuals to dataframe
PAIPR_df['res_accum'] = PAIPR_df['accum_2016']-PAIPR_df['accum_2011']
PAIPR_df['res_perc'] = (
    100*(PAIPR_df['res_accum'])
    /(PAIPR_df[['accum_2016','accum_2011']]).mean(axis=1))

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))

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

# %%[markdown]
# ## 2011 PAIPR-manual comparions
# 
# %%

# Import and format PAIPR results
dir1 = ROOT_DIR.joinpath('data/PAIPR-repeat/20111109/smb/')
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
dir_0 = ROOT_DIR.joinpath(
    'data/PAIPR-repeat/20111109/smb-manual/')
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

plt.rcParams.update({'font.size': 24})

def plot_TScomp(
    ts_df1, ts_df2, gdf_combo, labels, 
    yaxis=True, xlims=None, ylims=None,
    colors=['blue', 'red'], ts_err1=None, ts_err2=None):
    """This is a function to generate matplotlib objects that compare spatially overlapping accumulation time series.

    Args:
        ts_df1 (pandas.DataFrame): Dataframe containing time series for the first dataset.
        ts_df2 (pandas.DataFrame): Dataframe containing time series for the second dataset.
        gdf_combo (geopandas.geoDataFrame): Geodataframe with entries corresponding to the paired time series locations. Also contains a column 'Site' that groups the different time series according to their manual tracing location.
        labels (list of str): The labels used in the output plot to differentiate the time series dataframes.
        colors (list, optional): The colors to use when plotting the time series. Defaults to ['blue', 'red'].
        ts_err1 (pandas.DataFrame, optional): DataFrame containing time series errors corresponding to ts_df1. If "None" then the error is estimated from the standard deviations in annual results. Defaults to None.
        ts_err2 (pandas.DataFrame, optional): DataFrame containing time series errors corresponding to ts_df2. If "None" then the error is estimated from the standard deviations in annual results. Defaults to None.

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
        
        if xlims:
            axes[i].set_xlim(xlims)
        if ylims:
            axes[i].set_ylim(ylims)

        # Add legend and set title based on site name
        axes[i].grid(True)

        if i == 0:
            axes[i].legend()
        # axes[i].set_title('Site '+site+' time series')

        if not yaxis:
            axes[i].set_yticklabels([])
        else:
            axes[i].set_ylabel('Accum (mm/a)')

        if i==(len(site_list)-1):
            pass
        else:
            axes[i].set_xticklabels([])
            axes[i].set_xlabel(None)

    return fig

# %% 2011 comparison plots

# Generate indices corresponding to desired sites
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
    yaxis=False, xlims=[1990, 2010], ylims=[80, 700],
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
dir1 = ROOT_DIR.joinpath('data/PAIPR-repeat/20161109/smb/')
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
dir_0 = ROOT_DIR.joinpath(
    'data/PAIPR-repeat/20161109/smb-manual/')
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
    yaxis=False, xlims=[1990,2010], ylims=[80,700],
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

# Add residuals to dataframe
PAP_man_df['res_accum'] = PAP_man_df['accum_paipr']-PAP_man_df['accum_man']
PAP_man_df['res_perc'] = (
    100*(PAP_man_df['res_accum'])
    /(PAP_man_df[['accum_paipr','accum_man']]).mean(axis=1))

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
dir_0 = ROOT_DIR.joinpath(
    'data/PAIPR-repeat/20111109/smb-manual/')
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
dir_0 = ROOT_DIR.joinpath(
    'data/PAIPR-repeat/20161109/smb-manual/')
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
    yaxis=False, xlims=[1990,2010], ylims=[80,700],
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
idx_tmp2011 = gdf_MANtraces.query('Site != "B"')['ID_2011']
accum_tmp2011 = man2011_ALL.iloc[:,idx_tmp2011]
idx_tmp2016 = gdf_MANtraces.query('Site != "B"')['ID_2016']
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

# Add residuals to dataframe
man_df['res_accum'] = man_df['accum_2016']-man_df['accum_2011']
man_df['res_perc'] = (
    100*(man_df['res_accum'])
    /(man_df[['accum_2016','accum_2011']]).mean(axis=1))

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
    yaxis=True, xlims=[1990,2010], ylims=[80,700],
    labels=['2011 PAIPR', '2016 PAIPR'], 
    ts_err1=std_2011, ts_err2=std_2016)




# %%[markdown]
# ## Investigations of spatial variability
#  
# %% Spatial coherence and variability

def vario(
    points_gdf, lag_size, 
    d_metric='euclidean', vars='all', 
    stationarize=False, scale=True):
    """A function to calculate the experimental variogram for values associated with a geoDataFrame of points.

    Args:
        points_gdf (geopandas.geodataframe.GeoDataFrame): Location of points and values associated with those points to use for calculating distances and lagged semivariance.
        lag_size (int): The size of the lagging interval to use for binning semivariance values.
        d_metric (str, optional): The distance metric to use when calculating pairwise distances in the geoDataFrame. Defaults to 'euclidean'.
        vars (list of str, optional): The names of variables to use when calculating semivariance. Defaults to 'all'.
        scale (bool, optional): Whether to perform normalization (relative to max value) on semivariance values. Defaults to True.

    Returns:
        pandas.core.frame.DataFrame: The calculated semivariance values for each chosen input. Also includes the lag interval (index), the average separation distance (dist), and the number of paired points within each interval (cnt). 
    """

    # Get column names if 'all' is selected for "vars"
    if vars == "all":
        vars = points_gdf.drop(
            columns='geometry').columns

    # Extact trace coordinates and calculate pairwise distance
    locs_arr = np.array(
        [points_gdf.geometry.x, points_gdf.geometry.y]).T
    dist_arr = pdist(locs_arr, metric=d_metric)

    # Calculate the indices used for each pairwise calculation
    i_idx = np.empty(dist_arr.shape)
    j_idx = np.empty(dist_arr.shape)
    m = locs_arr.shape[0]
    for i in range(m):
        for j in range(m):
            if i < j < m:
                i_idx[m*i + j - ((i + 2)*(i + 1))//2] = i
                j_idx[m*i + j - ((i + 2)*(i + 1))//2] = j


    if stationarize:
        pass
    
    # Create dfs for paired-point values
    i_vals = points_gdf[vars].iloc[i_idx].reset_index(
        drop=True)
    j_vals = points_gdf[vars].iloc[j_idx].reset_index(
        drop=True)

    # Calculate squared difference bewteen variable values
    sqdiff_df = (i_vals - j_vals)**2
    sqdiff_df['dist'] = dist_arr

    # Create array of lag interval endpoints
    d_max = lag_size * (dist_arr.max() // lag_size + 1)
    lags = np.arange(0,d_max+1,lag_size)

    # Group variables based on lagged distance intervals
    df_groups = sqdiff_df.groupby(
        pd.cut(sqdiff_df['dist'], lags))

    # Calculate semivariance at each lag for each variable
    gamma_vals = (1/2)*df_groups[vars].mean()
    gamma_vals.index.name = 'lag'

    if scale:
        gamma_df = gamma_vals / gamma_vals.max()
    else:
        gamma_df = gamma_vals

    # Add distance, lag center, and count values to output
    gamma_df['dist'] = df_groups['dist'].mean()
    gamma_df['lag_cent'] = lags[1::]-lag_size//2
    gamma_df['cnt'] = df_groups['dist'].count()

    return gamma_df

# %%

# Calculate variogram for PAIPR results
gamma_df = vario(
    gdf_PAIPR, lag_size=200, vars=['accum_mu', 'accum_res'], 
    scale=True)

# Generate random noise with matched characteristics to PAIPR results
gdf_noise = gdf_PAIPR.copy()[
    ['geometry', 'accum_mu', 'accum_res']]
gdf_noise['accum_mu'] = np.random.normal(
    loc=gdf_noise['accum_mu'].mean(), 
    scale=gdf_noise['accum_mu'].std(), 
    size=gdf_noise.shape[0])
gdf_noise['accum_res'] = np.random.normal(
    loc=gdf_noise['accum_res'].mean(), 
    scale=gdf_noise['accum_res'].std(), 
    size=gdf_noise.shape[0])

# Calculate variogram for random noise results
gamma_noise = vario(
    gdf_noise, lag_size=200, vars=['accum_mu', 'accum_res'], 
    scale=True)

# %% Plots/exploration of semivariograms

# According to the discussion [here](https://stats.stackexchange.com/questions/361220/how-can-i-understand-these-variograms), I should limit my empirical variogram to no more than 1/2 my total domain
threshold = (3/5)*gamma_df['dist'].max()
data1 = gamma_df.query('dist <= @threshold')
data2 = gamma_noise.query('dist <= @threshold')

# Plots of mean accum and accum residuals variograms between PAIPR and noise
fig_var, axes = plt.subplots(1, 2, figsize=(24,9))
data1.plot(
    kind='scatter', ax=axes[0], color='blue', 
    x='lag_cent', y='accum_mu', label='Real data')
data2.plot(
    kind='scatter', ax=axes[0], color='red', 
    x='lag_cent', y='accum_mu', label='Noise')
axes[0].set_xlabel('Lag distance (m)')
axes[0].set_ylabel('Mean accum variogram')
data1.plot(
    kind='scatter', ax=axes[1], color='blue', 
    x='lag_cent', y='accum_res', label='Real data')
data2.plot(
    kind='scatter', ax=axes[1], color='red', 
    x='lag_cent', y='accum_res', label='Noise')
axes[1].set_xlabel('Lag distance (m)')
axes[1].set_ylabel('Accum residual variogram')

# %% Download missing REMA data

# Set REMA data directory
REMA_DIR = Path(
    '/media/durbank/WARP/Research/Antarctica/Data/DEMs/REMA')

# Import shapefile of DEM tile locations
dem_index = gpd.read_file(REMA_DIR.joinpath(
    'REMA_Tile_Index_Rel1.1/REMA_Tile_Index_Rel1.1.shp'))

# Keep only DEMs that contain accum traces
dem_index = (
    gpd.sjoin(dem_index, gdf_PAIPR, op='contains').iloc[
        :,0:dem_index.shape[1]]).drop_duplicates()

# Find and download missing REMA DSM tiles
tiles_list = pd.DataFrame(
    dem_index.drop(columns='geometry'))
get_REMA(tiles_list, REMA_DIR.joinpath('tiles_8m_v1.1'))

# %% Calculate slope/aspect for all REMA tiles

# Generate list of paths to downloaded DEMs required for topo calculations at given points
dem_list = [
    path for path 
    in REMA_DIR.joinpath("tiles_8m_v1.1").glob('**/*dem.tif') 
    if any(tile in str(path) for tile in dem_index.tile)]

# Calculate slope and aspect for each DEM (only ones not already present)
[calc_topo(dem) for dem in dem_list]

# %% Extract topo values and add to PAIPR gdf

# Extract elevation, slope, and aspect values for each trace 
# location
tif_dirs = [path.parent for path in dem_list]
for path in tif_dirs:
    gdf_PAIPR = topo_vals(
        path, gdf_PAIPR, slope=True, aspect=True)

# Set elev/slope/aspect to NaN for locations where elev<0
gdf_PAIPR.loc[gdf_PAIPR['elev']<0,'elev'] = np.nan
gdf_PAIPR.loc[gdf_PAIPR['elev']<0,'slope'] = np.nan
gdf_PAIPR.loc[gdf_PAIPR['elev']<0,'aspect'] = np.nan

# Remove extreme slope values
gdf_PAIPR.loc[gdf_PAIPR['slope']>60,'slope'] = np.nan
gdf_PAIPR.loc[gdf_PAIPR['slope']<0,'elev'] = np.nan
gdf_PAIPR.loc[gdf_PAIPR['slope']<0,'aspect'] = np.nan

# %% Extract flight parameters

def get_FlightData(flight_dir):
    """Function to extract OIB flight parameter data from .nc files and convert to geodataframe.

    Args:
        flight_dir (pathlib.PosixPath): The directory containing the .nc OIB files to extract and convert.

    Returns:
        geopandas.geodataframe.GeoDataFrame: Table of OIB flight parameters (altitude, heading, lat, lon, pitch, and roll) with their corresponding surface location and collection time.
    """
    # List of files to extract from
    nc_files = [file for file in flight_dir.glob('*.nc')]
    
    # Load as xarray dataset
    xr_flight = xr.open_dataset(
        nc_files.pop(0))[[
            'altitude', 'heading', 'lat', 
            'lon', 'pitch','roll']]

    # Concatenate data from all flights in directory
    for file in nc_files:

        flight_i = xr.open_dataset(file)[[
            'altitude', 'heading', 'lat', 
            'lon', 'pitch','roll']]
        xr_flight = xr.concat(
            [xr_flight, flight_i], dim='time')

    # Convert to dataframe and average values to ~200 m resolution
    flight_data = xr_flight.to_dataframe()
    flight_coarse = flight_data.rolling(
        window=40, min_periods=1).mean().iloc[0::40]

    # Convert to geodataframe in Antarctic coordinates
    gdf_flight = gpd.GeoDataFrame(
        data=flight_coarse.drop(columns=['lat', 'lon']), 
        geometry=gpd.points_from_xy(
            flight_coarse.lon, flight_coarse.lat), 
        crs='EPSG:4326').reset_index().to_crs(epsg=3031)

    return gdf_flight


def path_dist(locs):
    """Function to calculate the cummulative distance between points along a given path.

    Args:
        locs (numpy.ndarray): 2D array of points along the path to calculate distance.
    """
    # Preallocate array for distances between adjacent points
    dist_seg = np.empty(locs.shape[0])

    # Calculate the distances bewteen adjacent points in array
    for i in range(locs.shape[0]):
        if i == 0:
            dist_seg[i] = 0
        else:
            pos_0 = locs[i-1]
            pos = locs[i]
            dist_seg[i] = np.sqrt(
                (pos[0] - pos_0[0])**2 
                + (pos[1] - pos_0[1])**2)

    # Find cummulative path distance along given array
    dist_cum = np.cumsum(dist_seg)

    return dist_cum

#%%

# # Assign OIB flight data directory
# OIB_DIR = Path(
#     '/media/durbank/WARP/Research/Antarctica/Data/IceBridge/WAIS-central')
# gdf_flight2011 = get_FlightData(OIB_DIR.joinpath('20111109'))
# gdf_flight2016 = get_FlightData(OIB_DIR.joinpath('20161109'))

# # Save data for future use (cuts repetative computation time)
# gdf_flight2011.to_file(
#     ROOT_DIR.joinpath('data/flight_params/flight_20111109.geojson'), 
#     driver='GeoJSON')
# gdf_flight2016.to_file(
#     ROOT_DIR.joinpath('data/flight_params/flight_20161109.geojson'), 
#     driver='GeoJSON')

# Load flight parameter data
gdf_flight2011 = gpd.read_file(
    ROOT_DIR.joinpath('data/flight_params/flight_20111109.geojson'))
gdf_flight2016 = gpd.read_file(
    ROOT_DIR.joinpath('data/flight_params/flight_20161109.geojson'))

# locs_2011 = np.array([
#     gdf_flight2011.geometry.x, 
#     gdf_flight2011.geometry.y]).T

# dist_2011 = path_dist(locs_2011)

# %% Add intersecting flight parameter data to PAIPR gdf

# Find nearest neighbors between gdf_PAIPR and gdf_flight2011
dist_2011 = pd.DataFrame(nearest_neighbor(
    gdf_PAIPR, gdf_flight2011, 
    return_dist=True).drop(
        columns=['time', 'geometry'])).add_suffix('_res')
dist_2011['plane_elev'] = dist_2011['altitude_res']-gdf_PAIPR['elev']
# dist_2011 = nearest_neighbor(
#     gdf_PAIPR, gdf_flight2011, 
#     return_dist=True).drop(
#         columns=['time', 'geometry']).add_suffix('_2011')
# gdf_PAIPR = gdf_PAIPR.join(dist_2011)

# Find nearest neighbors between gdf_PAIPR and gdf_flight2016
dist_2016 = pd.DataFrame(nearest_neighbor(
    gdf_PAIPR, gdf_flight2016, 
    return_dist=True).drop(
        columns=['time', 'geometry'])).add_suffix('_res')
dist_2016['plane_elev'] = dist_2016['altitude_res']-gdf_PAIPR['elev']
# dist_2016 = nearest_neighbor(
#     gdf_PAIPR, gdf_flight2016, 
#     return_dist=True).drop(
#         columns=['time', 'geometry']).add_suffix('_2016')
# gdf_PAIPR = gdf_PAIPR.join(dist_2016)

plane_res = (dist_2016 - dist_2011).drop(
    columns=['altitude_res', 'distance_res'])

# Add column for absolute value of accum res
gdf_PAIPR['res_abs'] = np.abs(gdf_PAIPR['accum_res'])

# Join data to gdf_PAIPR
gdf_PAIPR = gdf_PAIPR.join(plane_res)

# %% Paired correlation plots for variables

# sns.pairplot(
#     data=gdf_PAIPR, 
#     vars=[
#         'accum_res', 'trace_dist', 'elev', 'slope', 'aspect', 'altitude_2011', 'heading_2011', 'pitch_2011', 'roll_2011', 
#         'altitude_2016', 'heading_2016', 'pitch_2016', 'roll_2016'], 
#     kind='kde', dropna=True)

sns.pairplot(
    data=gdf_PAIPR, 
    x_vars=[
        'accum_res', 'res_abs', 'accum_mu', 
        'trace_dist', 'elev', 'slope', 'aspect'],
    y_vars=['accum_res', 'res_abs', 'accum_mu'], 
    diag_kind='kde', dropna=True)

# sns.pairplot(
#     data=gdf_PAIPR, 
#     x_vars=['accum_res', 'res_abs', 
#         'plane_elev2011', 'heading_2011', 'pitch_2011', 'roll_2011'],
#         y_vars=['accum_res', 'res_abs'], 
#     diag_kind='kde', dropna=True)

sns.pairplot(
    data=gdf_PAIPR, 
    x_vars=[
        'accum_res', 'res_abs', 
        'plane_elev', 'heading_res', 'pitch_res', 'roll_res'], 
    y_vars=['accum_res', 'res_abs'], 
    diag_kind='kde', dropna=True)

# %%[markdown]
# ## Final figures used in article
# 
# %%

# PAIPR_df = PAIPR_df.query("Site != 'Null'")


PAP_man_df = PAP_man_df.query("Site != 'Null'")
man_df = man_df.query("Site != 'Null'")
# PAP_man_df = PAP_man_df.query("Site != 'B'")
# man_df = man_df.query("Site != 'B'")

df_2011 = PAP_man_df.query('flight==2011')
df_2016 = PAP_man_df.query('flight==2016')

# %% Density plots

plt.rcParams.update({'font.size': 22})
kde_fig = plt.figure(figsize=(12,8))

ax0 = kde_fig.add_subplot()
ax0.axvline(color='black', linestyle='--')
ax1 = kde_fig.add_subplot()
sns.kdeplot(
    ax=ax1, 
    data=PAIPR_df['res_perc'], 
    label='2016-2011 PAIPR', linewidth=4, color='#d55e00')
ax2 = kde_fig.add_subplot()
sns.kdeplot(
    ax=ax2, 
    data=df_2011['res_perc'], 
    label='2011 PAIPR-manual', linewidth=4, color='#cc79a7')   
ax3 = kde_fig.add_subplot()
sns.kdeplot(
    ax=ax3, 
    data=df_2016['res_perc'], 
    label='2016 PAIPR-manual', linewidth=4, color='#0072b2')
ax4 = kde_fig.add_subplot()
sns.kdeplot(
    ax=ax4, 
    data=man_df['res_perc'], 
    label='2016-2011 manual', linewidth=4, color='#009e73')
ax1.legend()
ax1.set_xlabel('% Bias')


# %% Paired t tests for PAIPR

from scipy import stats

val_bias = []
t_stat = []
t_crit = []
p_val = []
moe_val = []
for site in np.unique(PAIPR_df.Site):

    data_i = PAIPR_df.query('Site == @site')
    SE_i = data_i['res_perc'].std()/np.sqrt(data_i.shape[0])
    t_i = abs(data_i['res_perc'].mean()/SE_i)
    crit_i = stats.t.ppf(q=0.995, df=data_i.shape[0]-1)

    val_bias.append(data_i['res_perc'].mean())
    t_stat.append(t_i)
    t_crit.append(crit_i)
    p_val.append(2*(1-stats.t.cdf(x=t_i, df=data_i.shape[0]-1)))
    moe_val.append(crit_i*SE_i)

    print(f"For Site {site}, the mean bias is {data_i['res_perc'].mean():.1f}(+/-){crit_i*SE_i:.2f}")

print(p_val)
print('As all p-vals are 0, for all sites 2011 results are statistically different from 2016 results :(')

# %% Paired t tests for PAIPR (averaged by year)

from scipy import stats

val_bias = []
t_stat = []
t_crit = []
p_val = []
moe_val = []
for site in np.unique(PAIPR_df.Site):

    data_i = PAIPR_df.query('Site == @site').groupby('Year').mean()
    SE_i = data_i['res_perc'].std()/np.sqrt(data_i.shape[0])
    t_i = abs(data_i['res_perc'].mean()/SE_i)
    crit_i = stats.t.ppf(q=0.995, df=data_i.shape[0]-1)

    val_bias.append(data_i['res_perc'].mean())
    t_stat.append(t_i)
    t_crit.append(crit_i)
    p_val.append(2*(1-stats.t.cdf(x=t_i, df=data_i.shape[0]-1)))
    moe_val.append(crit_i*SE_i)

    print(f"For Site {site}, the mean bias is {data_i['res_perc'].mean():.1f}(+/-){crit_i*SE_i:.2f}")

print(p_val)

# %% Paired t tests for manual

from scipy import stats

val_bias = []
t_stat = []
t_crit = []
p_val = []
moe_val = []
for site in np.unique(man_df.Site):

    data_i = man_df.query('Site == @site')
    SE_i = data_i['res_perc'].std()/np.sqrt(data_i.shape[0])
    t_i = abs(data_i['res_perc'].mean()/SE_i)
    crit_i = stats.t.ppf(q=0.995, df=data_i.shape[0]-1)

    val_bias.append(data_i['res_perc'].mean())
    t_stat.append(t_i)
    t_crit.append(crit_i)
    p_val.append(2*(1-stats.t.cdf(x=t_i, df=data_i.shape[0]-1)))
    moe_val.append(crit_i*SE_i)

    print(f"For Site {site}, the mean bias is {data_i['res_perc'].mean():.1f}(+/-){crit_i*SE_i:.2f}")

print(p_val)

# %%

print(
    f"The mean bias between PAIPR-derived results "
    f"between 2016 and 2011 flights is "
    f"{PAIPR_df['res_accum'].mean():.1f} mm/yr ("
    f"{PAIPR_df['res_perc'].mean():.2f}% of mean accum) "
    f"with a RMSE of {PAIPR_df['res_perc'].std():.2f}%."
)
# print(
#     f"The mean annual accumulation for PAIPR results are "
#     f"{PAIPR_df['accum_2011'].mean():.0f} mm/yr for 2011 " 
#     f"and {PAIPR_df['accum_2016'].mean():.0f} mm/yr for 2016"
# )
# print(
#     f"The mean standard deviations of the annual "
#     f"accumulation estimates are "
#     f"{(100*PAIPR_df['std_2011']/PAIPR_df['accum_2011']).mean():.2f}% " 
#     f"for the 2011 flight and "
#     f"{(100*PAIPR_df['std_2016']/PAIPR_df['accum_2016']).mean():.2f}% "
#     f"for the 2016 flight."
# )

print(
    f"The mean bias between manually-derived results "
    f"between 2016 and 2011 flights is "
    f"{man_df['res_accum'].mean():.1f} mm/yr ("
    f"{man_df['res_perc'].mean():.2f}% of mean accum) "
    f"with a RMSE of {man_df['res_perc'].std():.2f}%."
)
# print(
#     f"The mean annual accumulation for manual results are "
#     f"{man_df['accum_2011'].mean():.0f} mm/yr for 2011 " 
#     f"and {man_df['accum_2016'].mean():.0f} mm/yr for 2016"
# )
# print(
#     f"The mean standard deviations of the manual annual "
#     f"accumulation estimates are "
#     f"{(100*man_df['std_2011']/man_df['accum_2011']).mean():.2f}% " 
#     f"for the 2011 flight and "
#     f"{(100*man_df['std_2016']/man_df['accum_2016']).mean():.2f}% "
#     f"for the 2016 flight."
# )

print(
    f"The mean bias between manually-traced and "
    f"PAIPR-derived accumulation for 2011 is "
    f"{df_2011['res_accum'].mean():.2f} mm/yr ("
    f"{df_2011['res_perc'].mean():.2f}% of the mean accumulation) "
    f"with a RMSE of {df_2011['res_perc'].std():.2f}%"
)
print(
    f"The mean annual accumulation for 2011 results are "
    f"{df_2011['accum_paipr'].mean():.0f} mm/yr for PAIPR " 
    f"and {df_2011['accum_man'].mean():.0f} mm/yr "
    f"for manual results."
)
print(
    f"The mean standard deviations of the annual "
    f"accumulation estimates are {(df_2011['std_man']/df_2011['accum_man']).mean()*100:.2f}% " 
    f"for 2011 manual layers and "
    f"{(df_2011['std_paipr']/df_2011['accum_paipr']).mean()*100:.2f}% "
    f"for 2011 PAIPR layers."
)

print(
    f"The mean bias between manually-traced and "
    f"PAIPR-derived accumulation for 2016 is "
    f"{df_2016['res_accum'].mean():.2f} mm/yr ("
    f"{df_2016['res_perc'].mean():.2f}% of the mean accumulation) "
    f"with a RMSE of {df_2016['res_perc'].std():.2f}%"
)
print(
    f"The mean annual accumulation for 2016 results are "
    f"{df_2016['accum_paipr'].mean():.0f} mm/yr for PAIPR " 
    f"and {df_2016['accum_man'].mean():.0f} mm/yr "
    f"for manual results."
)
print(
    f"The mean standard deviations of the annual "
    f"accumulation estimates are {(df_2016['std_man']/df_2016['accum_man']).mean()*100:.2f}% " 
    f"for 2016 manual layers and "
    f"{(df_2016['std_paipr']/df_2016['accum_paipr']).mean()*100:.2f}% "
    f"for 2016 PAIPR layers."
)

# %%

# print(
#     f"The mean bias between PAIPR-derived results "
#     f"between 2016 and 2011 flights is "
#     f"{res_PAIPR.values.mean():.1f} mm/yr ("
#     f"{resPAIPR_perc.values.mean():.2f}% of mean accum) "
#     f"with a RMSE of {resPAIPR_perc.values.std():.2f}%."
# )
# print(
#     f"The mean annual accumulation for PAIPR results are "
#     f"{accum_2011.values.mean():.0f} mm/yr for 2011 " 
#     f"and {accum_2016.values.mean():.0f} mm/yr for 2016"
# )
# print(
#     f"The mean standard deviations of the annual "
#     f"accumulation estimates are "
#     f"{(std_2011.values/accum_2011.values).mean()*100:.2f}% " 
#     f"for the 2011 flight and "
#     f"{(std_2016.values/accum_2016.values).mean()*100:.2f}% "
#     f"for the 2016 flight."
# )

# print(
#     f"The mean bias between manually-traced and "
#     f"PAIPR-derived accumulation for 2011 is "
#     f"{res_2011.values.mean():.2f} mm/yr ("
#     f"{res2011_perc.values.mean():.2f}% of the mean accumulation) "
#     f"with a RMSE of {res2011_perc.values.std():.2f}%"
# )
# print(
#     f"The mean annual accumulation for 2011 results are "
#     f"{Maccum_2011.values.mean():.0f} mm/yr for PAIPR " 
#     f"and {man2011_accum.values.mean():.0f} mm/yr "
#     f"for manual results."
# )
# print(
#     f"The mean standard deviations of the annual "
#     f"accumulation estimates are {(man2011_std/man2011_accum).values.mean()*100:.2f}% " 
#     f"for 2011 manual layers and "
#     f"{(Mstd_2011/Maccum_2011).values.mean()*100:.2f}% "
#     f"for 2011 PAIPR layers."
# )

# print(
#     f"The mean bias between manually-traced and "
#     f"PAIPR-derived accumulation for 2016 is "
#     f"{res_2016.values.mean():.2f} mm/yr ("
#     f"{res2016_perc.values.mean():.2f}% of mean accum) "
#     f"with a RMSE of {res2016_perc.values.std():.2f}%."
# )
# print(
#     f"The mean annual accumulation for 2016 results are "
#     f"{Maccum_2016.values.mean():.0f} mm/yr for PAIPR " 
#     f"and {man2016_accum.values.mean():.0f} mm/yr "
#     f"for manual results."
# )
# print(
#     f"The mean standard deviations of the annual "
#     f"accumulation estimates are "
#     f"{(man2016_std/man2016_accum).values.mean()*100:.2f}% " 
#     f"for 2016 manual layers and "
#     f"{(Mstd_2016/Maccum_2016).values.mean()*100:.2f}% "
#     f"for 2016 PAIPR layers."
# )

# print(
#     f"The mean bias in manually-derived annual accumulation 2016 vs 2011 flights is "
#     f"{man_res.values.mean():.1f} mm/yr ("
#     f"{man_res_perc.values.mean():.2f}% of mean accum) "
#     f"with a RMSE of {man_res_perc.values.std():.2f}%."
# )
# print(
#     f"The mean annual accumulation for manual results are "
#     f"{accum_man2011.values.mean():.0f} mm/yr for 2011 " 
#     f"and {accum_man2016.values.mean():.0f} mm/yr for 2016"
# )
# print(
#     f"The mean standard deviations of the annual accumulation estimates are {(std_man2011/accum_man2011).values.mean()*100:.2f}% for 2011 and {(std_man2016/accum_man2016).values.mean()*100:.2f}% for 2016.")

# print(
#     f"The mean bias in manually-derived annual accumulation 2016 vs 2011 flights is "
#     f"{res_tmp.values.mean():.1f} mm/yr ("
#     f"{res_perc_tmp.values.mean():.2f}% of mean accum) "
#     f"with a RMSE of {res_perc_tmp.values.std():.2f}%."
# )
# print(
#     f"The mean annual accumulation for manual results are "
#     f"{accum_tmp2011.values.mean():.0f} mm/yr for 2011 " 
#     f"and {accum_tmp2016.values.mean():.0f} mm/yr for 2016"
# )
# print(
#     f"The mean standard deviations of the annual accumulation estimates are {(std_man2011/accum_man2011).values.mean()*100:.2f}% for 2011 and {(std_man2016/accum_man2016).values.mean()*100:.2f}% for 2016.")

# %%

def panels_121(
    datum, 
    x_vars=[
        'accum_2011','accum_2011','accum_man','accum_man'],
    y_vars=[
        'accum_2016','accum_2016','accum_paipr','accum_paipr'],
    TOP=False, BOTTOM=False, xlabels=None, ylabels=None, 
    kde_colors=['#d55e00', '#cc79a7', '#0072b2', '#009e73'],
    kde_labels=[
        '2016-2011 PAIPR', '2016-2011 manual', 
        '2011 PAIPR-manual', '2016 PAIPR-manual'], 
    plot_min=None, plot_max=None, size=500):
    """A function to generate a Holoviews Layout consisting of 1:1 plots of the given datum, with a kernel density plot also showing the residuals between the given x and y variables.

    Args:
        datum (list of pandas.DataFrame): First data group to include for generating plots. Variables in dataframe must include an x-variable, a y-variable, and a 'Year' variable.
        x_vars (list of str, optional): The names of the x-variables included in the datum. Defaults to [ 'accum_2011','accum_2011','accum_man','accum_man'].
        y_vars (list of str, optional): The names of the y-variables included in the datum.. Defaults to [ 'accum_2016','accum_2016','accum_paipr','accum_paipr'].
        TOP (bool, optional): Whether the returned Layout will be at the top of a Layout stack. Defaults to False.
        BOTTOM (bool, optional): Whether the returned Layout will be at the bottom of a Layout stack. Defaults to False.
        xlabels ([type], optional): List of strings to use as the x-labels for the 1:1 plots. Defaults to None.
        ylabels (list, optional): List of strings to use as the y-labels for the 1:1 plots. Defaults to None.
        kde_colors (list, optional): List of colors to use in kde plots. Defaults to ['blue', 'red', 'purple', 'orange'].
        kde_labels (list, optional): List of labels to use in kde plots. Defaults to [ '2016-2011 PAIPR', '2016-2011 manual', '2011 PAIPR-manual', '2016 PAIPR-manual'].
        plot_min (float, optional): Specify a lower bound to the generated plots. Defaults to None.
        plot_max (float, optional): Specifies an upper bound to the generated plots. Defaults to None.
        size (int, optional): The output size (both width and height) in pixels of individual subplots in the Layout panel. Defaults to 500.
        # font_scaler (float, optional): How much to scale the generated plot text. Defaults to no additional scaling.

    Returns:
        holoviews.core.layout.Layout: Figure Layout consisting of multiple 1:1 subplots and a subplot with kernel density estimates of the residuals of the various datum.
    """

    # Preallocate subplot lists
    one2one_plts = []
    kde_plots = []
    if xlabels is None:
        xlabels = np.repeat(None, len(datum))
    if ylabels is None:
        ylabels = np.repeat(None, len(datum))

    for i, data in enumerate(datum):

        # Get names of x-y variables for plotting
        x_var = x_vars[i]
        y_var = y_vars[i]

        # Get global axis bounds and generate 1:1 line
        if plot_min is None:
            plot_min = np.min(
                [data[x_var].min(), data[y_var].min()])
        if plot_max is None:
            plot_max = np.max(
                [data[x_var].max(), data[y_var].max()])
        one2one_line = hv.Curve(data=pd.DataFrame(
            {'x':[plot_min, plot_max], 
            'y':[plot_min, plot_max]})).opts(color='black')

        # Generate 1:1 scatter plot, gradated by year
        scatt_yr = hv.Points(
            data=data, kdims=[x_var, y_var], 
            vdims=['Year']).opts(
                xlim=(plot_min, plot_max), 
                ylim=(plot_min, plot_max), 
                xlabel=xlabels[i], 
                ylabel=ylabels[i], 
                color='Year', cmap='Category20b', colorbar=False)
        scatt_yr.opts(
            labelled=[], show_grid=True, xaxis=None, yaxis=None)

        # Special formatting given position of subplot in figure
        if i==0:
            scatt_yr.opts(yaxis='left')
            # scatt_yr.opts(ylabel='Annual accum (mm/yr)')
        if i==len(datum)-1:
            scatt_yr.opts(colorbar=True)
        if TOP:
            scatt_yr.opts(xaxis='top')
        if BOTTOM:
            scatt_yr.opts(xaxis='bottom')

        # Combine 1:1 line and scatter plot and add to plot list
        one2one_plt = (one2one_line * scatt_yr).opts(
            width=size, height=size, 
            # fontscale=3,
            fontsize={'ticks':20, 'xticks':30, 'yticks':30}, 
            xrotation=90, 
            xticks=4, yticks=4)
        if i==len(datum)-1:
            one2one_plt.opts(width=int(size+0.10*size))
        one2one_plts.append(one2one_plt)

        # Generate kde for residuals of given estimates and 
        # add to plot list 
        kde_data = (
            100*(data[y_var]-data[x_var]) 
            / data[[x_var, y_var]].mean(axis=1))
        kde_plot = hv.Distribution(
            kde_data, label=kde_labels[i]).opts(
            filled=False, line_color=kde_colors[i], 
            line_width=5)
        kde_plots.append(kde_plot)

    # Generate and decorate combined density subplot
    fig_kde = (
        kde_plots[0] * kde_plots[1] 
        * kde_plots[2] * kde_plots[3]).opts(
            xaxis=None, yaxis=None, 
            width=size, height=size, show_grid=True
        )
    
    # Specific formatting based on subplot position
    if TOP:
        fig_kde = fig_kde.opts(
            xaxis='top', xlabel='% Bias', xrotation=90, 
            fontsize={'legend':25, 'xticks':18, 'xlabel':22})
    elif BOTTOM:
        fig_kde = fig_kde.opts(
            xaxis='bottom', xlabel='% Bias', xrotation=90,
            fontsize={'legend':25, 'xticks':18, 'xlabel':22})
    else:
        fig_kde = fig_kde.opts(show_legend=False)


    # Generate final panel Layout with 1:1 and kde plots
    fig_layout = hv.Layout(
        one2one_plts[0] + one2one_plts[1] 
        + one2one_plts[2] + one2one_plts[3] 
        + fig_kde).cols(5)
    
    return fig_layout

# %%

# Preallocate Layout list
figs_supp2 = []

# Add full dataset Layout to Layout list
figs_supp2.append(
    panels_121(
        datum=[PAIPR_df, man_df, df_2011, df_2016],
        plot_min=80, plot_max=820, TOP=True))

# Get list of unique sites in data
site_list = np.unique(gdf_PAIPR['Site']).tolist()
site_list.remove("Null")

# Iterate through sites
for i, site in enumerate(site_list):


    paipr_i = PAIPR_df.query("Site == @site")
    man_i = man_df.query("Site == @site")
    df2011_i = df_2011.query("Site == @site")
    df2016_i = df_2016.query("Site == @site")

    data_i = [paipr_i, man_i, df2011_i, df2016_i]

    # Generate panel figures for current site and add 
    # to figure list
    if i==(len(site_list)-1):
        figs_supp2.append(panels_121(
            datum=data_i,  plot_min=80, 
            plot_max=820, BOTTOM=True))
    else:
        figs_supp2.append(panels_121(
            datum=data_i, plot_min=80, plot_max=820))

# Combine individual site panels to final Supplementary figure
fig_supp2 = hv.Layout(
    figs_supp2[0] + figs_supp2[1] + figs_supp2[2] 
    + figs_supp2[3] + figs_supp2[4] + figs_supp2[5]
    + figs_supp2[6]).cols(5)

# %% Data location map

data_map = (
    # elev_plt.opts(colorbar=False)
    hill_plt
    * cont_plt.opts(colorbar=False)
    * plt_locCOMB * plt_manPTS 
    * (plt_accum2011 * plt_accum2016) 
    * plt_labels.opts(text_font_size='36pt')
    ).opts(fontscale=3, width=1200, height=1200)

res_map = (
    # elev_plt.opts(colorbar=False) 
    hill_plt * 
    cont_plt.opts(colorbar=False)
    * plt_manPTS* plt_res 
    * plt_labels.opts(text_font_size='36pt')
    ).opts(
        ylim=(gdf_bounds['y_range'][0], -2.25E5), 
        fontscale=3, width=1200, height=715)

# %%

tsfig_PAIPR.savefig(
    fname=ROOT_DIR.joinpath(
        'docs/Figures/oib-repeat/tsfig_PAIPR.svg'))
tsfig_manual.savefig(
    fname=ROOT_DIR.joinpath(
        'docs/Figures/oib-repeat/tsfig_man.svg'))
tsfig_2011.savefig(
    fname=ROOT_DIR.joinpath(
        'docs/Figures/oib-repeat/tsfig_2011.svg'))
tsfig_2016.savefig(
    fname=ROOT_DIR.joinpath(
        'docs/Figures/oib-repeat/tsfig_2016.svg'))

kde_fig.savefig(fname=ROOT_DIR.joinpath(
    'docs/Figures/oib-repeat/kde_fig.svg'))

hv.save(data_map, ROOT_DIR.joinpath(
    'docs/Figures/oib-repeat/data_map.png'))
hv.save(res_map, ROOT_DIR.joinpath(
    'docs/Figures/oib-repeat/res_map.png'))
hv.save(fig_supp2, ROOT_DIR.joinpath(
    'docs/Figures/oib-repeat/one2one_plts.png'))

# %%

cont_bar = cont_plt.opts(
    colorbar=True, fontscale=3, width=1200, height=1200)
hv.save(cont_bar, ROOT_DIR.joinpath(
    'docs/Figures/oib-repeat/cont_bar.png'))

# %% Example echogram figures

echo2011 = xr.load_dataset(ROOT_DIR.joinpath(
    'data/oib-examples/IRSNO1B_20111109_02_255.nc'))[
    ['amplitude', 'lat', 'lon', 'Surface', 'fasttime']].transpose()

# Calculate distance along time dimension and add to ds
tmp = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(
        echo2011['lon'].data, echo2011['lat'].data, 
        crs='epsg:4326'))
tmp.to_crs(epsg=3031, inplace=True)
echo2011['distance'] = (
    ('time'), 
    np.flip(path_dist(
        np.array([tmp.geometry.x, tmp.geometry.y]).T)))

# Set new coordinates
echo2011 = echo2011.set_coords(['fasttime', 'distance'])

# echo2011 = echo2011.where(
#     (echo2011.fasttime<=np.timedelta64(3070, 'ns')) 
#     & (echo2011.fasttime>=np.timedelta64(2900, 'ns')), 
#     drop=True)

# echo_hv = hv.Dataset(
#     (echo2011['distance'].data, 
#     echo2011['fasttime'].data.astype('float'), 
#     echo2011['amplitude'].data), 
#     ['distance', 'fasttime'], 'amplitude')

# hv.Image(echo_hv).opts(
#     invert_yaxis=True, cmap='bone', width=900, height=600)


# %%

amp_data = echo2011['amplitude'].data
time_delay = pd.DataFrame({'fasttime':echo2011['fasttime'].data})
surf = echo2011['Surface'].data

v_depth = 1000
clip_data = np.empty((v_depth, amp_data.shape[1]))
time_data = np.empty((v_depth, amp_data.shape[1]), dtype='<m8[ns]')
v_offset = np.timedelta64(75, 'ns')
for i in range(amp_data.shape[1]):
    col = amp_data[:,i]
    start_idx = time_delay[
        time_delay['fasttime'].gt((surf[i]-v_offset))].index[0]
    time_data[:,i] = time_delay.loc[
        start_idx:start_idx+v_depth-1, 'fasttime'].values
    clip_data[:,i] = col[start_idx:start_idx+v_depth]

depth = np.linspace(-0.5, 6, num=v_depth)




echo_hv = hv.Dataset(
    (echo2011['distance'].data, 
    depth, 
    clip_data), 
    ['distance', 'depth'], 'amplitude')

hv.Image(echo_hv).opts(
    invert_yaxis=True, cmap='bone', width=900, height=600)


# %%[markdown]
# ## Investigations of depth variability and QC effect
# 
# %%

# Import current 20111109 results
dir_ = ROOT_DIR.joinpath('data/PAIPR-repeat/20111109/smb/')
data_raw = import_PAIPR(dir_)
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)
smb_2011 = format_PAIPR(
    data_0, start_yr=1990, end_yr=2010).drop(
    'elev', axis=1).set_index(['collect_time', 'Year'])

# Import 20111109 depth results
dir_ = ROOT_DIR.joinpath('data/PAIPR-repeat/20111109/depth/')
depth_2011 = import_PAIPR(dir_).rename(
    columns={"IHR_year":"Year"}).set_index(
        ['collect_time', 'Year'])

# Join 2011 dataframes and extract gdf locations
df_depths2011 = smb_2011.join(
    depth_2011.drop(columns=['Lat', 'Lon']), 
    how='inner').reset_index().groupby(
        ['trace_ID', 'Year']).mean()
df_grouped2011 = df_depths2011[['Lat', 'Lon']].groupby(
        'trace_ID').mean().reset_index()
gdf_depth2011 = gpd.GeoDataFrame(
    data=df_grouped2011.drop(columns=['Lat', 'Lon']), 
    geometry=gpd.points_from_xy(
        x=df_grouped2011.Lon, y=df_grouped2011.Lat), 
    crs="EPSG:4326")
gdf_depth2011.to_crs(epsg=3031, inplace=True)

# ## Repeat for 2016 data

# Import current 20161109 results
dir_ = ROOT_DIR.joinpath('data/PAIPR-repeat/20161109/smb/')
data_raw = import_PAIPR(dir_)
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)
smb_2016 = format_PAIPR(
    data_0, start_yr=1990, end_yr=2010).drop(
    'elev', axis=1).set_index(['collect_time', 'Year'])

# Import 20161109 depth results
dir_ = ROOT_DIR.joinpath('data/PAIPR-repeat/20161109/depth/')
depth_2016 = import_PAIPR(dir_).rename(
    columns={"IHR_year":"Year"}).set_index(
        ['collect_time', 'Year'])

# Join 2016 dataframes and extract gdf locations
df_depths2016 = smb_2016.join(
    depth_2016.drop(columns=['Lat', 'Lon']), 
    how='inner').reset_index().groupby(
        ['trace_ID', 'Year']).mean()
df_grouped2016 = df_depths2016[['Lat', 'Lon']].groupby(
        'trace_ID').mean().reset_index()
gdf_depth2016 = gpd.GeoDataFrame(
    data=df_grouped2016.drop(columns=['Lat', 'Lon']), 
    geometry=gpd.points_from_xy(
        x=df_grouped2016.Lon, y=df_grouped2016.Lat), 
    crs="EPSG:4326")
gdf_depth2016.to_crs(epsg=3031, inplace=True)


# Find nearest neighbors between 2011 and 2016 
# (within 500 m)
df_dist = nearest_neighbor(
    gdf_depth2011, gdf_depth2016, return_dist=True)
idx_paipr = df_dist['distance'] <= 250
dist_overlap = df_dist[idx_paipr]


yr_vals = np.arange(
    df_depths2011.index.get_level_values(1).min(), 
    df_depths2011.index.get_level_values(1).max()+1)


ID_2011 = gdf_depth2011.loc[dist_overlap.index,'trace_ID'].values
midx_2011 = pd.MultiIndex.from_arrays([
    np.repeat(ID_2011, yr_vals.shape[0]), 
    np.tile(yr_vals, ID_2011.shape[0])], 
    names=['trace_ID', 'Year'])
depths_2011 = df_depths2011.loc[midx_2011]

ID_2016 = dist_overlap['trace_ID'].values
midx_2016 = pd.MultiIndex.from_arrays([
    np.repeat(ID_2016, yr_vals.shape[0]), 
    np.tile(yr_vals, ID_2016.shape[0])], 
    names=['trace_ID', 'Year'])
depths_2016 = df_depths2016.loc[midx_2016]


# Create new combined df for investigating depth differences
df_depths = pd.DataFrame(
    {'ID_new':np.repeat(
        np.arange(depths_2011.shape[0]/yr_vals.shape[0]), 
        yr_vals.shape[0]), 
    'ID_2011':depths_2011.index.get_level_values(0), 
    'ID_2016':depths_2016.index.get_level_values(0), 
    'Year':depths_2011.index.get_level_values(1), 
    'Lat':depths_2011['Lat'].values, 
    'Lon':depths_2011['Lon'].values, 
    'elev_2011':depths_2011['elev'].values, 
    'elev_2016':depths_2016['elev'].values, 
    'depth_2011':depths_2011['IHR_depth'].values, 
    'depth_2016':depths_2016['IHR_depth'].values})
df_depths['depth_res'] = (
    df_depths['depth_2016'] - df_depths['depth_2011'])

# %% 

hv.Distribution(df_depths, kdims=['depth_res'], vdims=['Year']).groupby('Year')


# %% Compare PAIPR results from different random seeds

# Import old 20111109 results
dir_old = ROOT_DIR.joinpath('data/PAIPR-outputs/20111109')
data_old = import_PAIPR(dir_old)
data_old.query('QC_flag != 2', inplace=True)
smb_old = data_old.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)
data_test = format_PAIPR(
    smb_old, start_yr=1990, end_yr=2010).drop(
    'elev', axis=1)
aTest_ALL = data_test.pivot(
    index='Year', columns='trace_ID', values='accum')
stdTest_ALL = data_test.pivot(
    index='Year', columns='trace_ID', values='std')

# Import current 20111109 results
dir1 = ROOT_DIR.joinpath('data/PAIPR-repeat/20111109/smb/')
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

# Create gdf of mean results for each trace and 
# transform to Antarctic Polar Stereographic
gdf_TestData = long2gdf(data_test)
gdf_TestData.to_crs(epsg=3031, inplace=True)
gdf_2011 = long2gdf(data_2011)
gdf_2011.to_crs(epsg=3031, inplace=True)

df_dist = nearest_neighbor(
    gdf_2011, gdf_TestData, return_dist=True)
idx_paipr = df_dist['distance'] <= 100
dist_overlap = df_dist[idx_paipr]

# Create numpy arrays for relevant results
accum_2011_test = a2011_ALL.iloc[
    :,dist_overlap.index]
std_2011_test = std2011_ALL.iloc[
    :,dist_overlap.index]
accum_test = aTest_ALL.iloc[
    :,dist_overlap['trace_ID']]
std_test = stdTest_ALL.iloc[
    :,dist_overlap['trace_ID']]

# Create new gdf of subsetted results
gdf_test = gpd.GeoDataFrame(
    {'ID_2011': dist_overlap.index.values, 
    'ID_test': dist_overlap['trace_ID'].values, 
    'trace_dist': dist_overlap['distance'].values,
    'accum_2011': 
        accum_2011_test.mean(axis=0).values, 
    'accum_test': 
        accum_test.mean(axis=0).values},
    geometry=dist_overlap.geometry.values)

# Calculate bulk accum mean and accum residual
gdf_test['accum_mu'] = gdf_test[
    ['accum_2011', 'accum_test']].mean(axis=1)
gdf_test['accum_res'] = (
    (gdf_test.accum_test - gdf_test.accum_2011) 
    / gdf_test.accum_mu)

gv.Points(
    gdf_test, vdims=['accum_mu', 'accum_res'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='accum_res', size=5, 
        cmap='coolwarm_r', colorbar=True, symmetric=True, 
        tools=['hover'], height=750, width=750)





# Generate indices corresponding to desired sites
gdf_test['Site'] = np.repeat(
    'Null', gdf_test.shape[0])
for label in chunk_centers['Site']:

    geom = chunk_centers.query('Site == @label').geometry
    idx = (gpd.GeoSeries(
        data=gpd.points_from_xy(
            np.repeat(geom.x, gdf_test.shape[0]), 
            np.repeat(geom.y, gdf_test.shape[0])), 
        crs=gdf_test.crs).distance(
            gdf_test.reset_index()) <= 30000).values
    
    gdf_test.loc[idx,'Site'] = label

# Create dataframes for scatter plots
PAIPR_df = pd.DataFrame(
    {'tmp_ID': np.tile(
        np.arange(0,accum_2011_test.shape[1]), 
        accum_2011_test.shape[0]), 
    'Site': np.tile(
        gdf_test['Site'], accum_2011_test.shape[0]), 
    'Year': np.reshape(
        np.repeat(
            accum_2011_test.index, accum_2011_test.shape[1]), 
        accum_2011_test.size), 
    'accum_2011': 
        np.reshape(
            accum_2011_test.values, accum_2011_test.size), 
    'std_2011': np.reshape(
        std_2011_test.values, std_2011_test.size), 
    'accum_test': np.reshape(
        accum_test.values, accum_test.size), 
    'std_test': np.reshape(
        std_test.values, std_test.size)})

# Add residuals to dataframe
PAIPR_df['res_accum'] = PAIPR_df['accum_test']-PAIPR_df['accum_2011']
PAIPR_df['res_perc'] = (
    100*(PAIPR_df['res_accum'])
    /(PAIPR_df[['accum_test','accum_2011']]).mean(axis=1))

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))

scatt_yr = hv.Points(
    data=PAIPR_df, 
    kdims=['accum_2011', 'accum_test'], 
    vdims=['Year'])
    
test_1to1_plt = one_to_one.opts(color='black')*scatt_yr.opts(
    xlim=(100,750), ylim=(100,750), 
    xlabel='2011 PAIPR (mm/yr)', 
    ylabel='Test PAIPR (mm/yr)', 
    color='Year', cmap='plasma', colorbar=True, 
    width=600, height=600, fontscale=1.75)

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=PAIPR_df, 
    kdims=['accum_2011', 'accum_test'], 
    vdims=['Year', 'Site']).groupby('Site')
test_res_plt = one_to_one.opts(color='black')*scatt_yr.opts(
    xlim=(100,750), ylim=(100,750), 
    xlabel='2011 flight (mm/yr)', 
    ylabel='Test flight (mm/yr)', 
    color='Year', cmap='plasma', colorbar=True, 
    width=600, height=600, fontscale=1.75)

test_1to1_plt + site_res_plt

 # %%[markdown]
# ## Matplotlib versions of final figures (WIP)
#  
# %% matplotlib versions

# plt_accum2011 = gv.Points(
#     gdf_2011, crs=ANT_proj, vdims=['accum']).opts(
#         projection=ANT_proj, color='accum', 
#         cmap='viridis', colorbar=True, s=50)
# plt_accum2016 = gv.Points(
#     gdf_2016, crs=ANT_proj, vdims=['accum']).opts(
#         projection=ANT_proj, color='accum', 
#         cmap='viridis', colorbar=True, s=50)

# # plt_loc2011 = gv.Points(
# #     gdf_2011, crs=ANT_proj, vdims=['accum','std']).opts(
# #         projection=ANT_proj, color='blue', alpha=0.9, 
# #         s=25)
# # plt_loc2016 = gv.Points(
# #     gdf_2016, crs=ANT_proj, vdims=['accum','std']).opts(
# #         projection=ANT_proj, color='red', alpha=0.9, 
# #         s=25)
# plt_locCOMB = gv.Points(
#     gdf_PAIPR, crs=ANT_proj, 
#     vdims=['accum_2011','accum_2016']).opts(
#         projection=ANT_proj, color='orange', 
#         s=200)



# plt_manPTS = gv.Points(
#     chunk_centers, crs=ANT_proj, 
#     vdims='Site').opts(
#         projection=ANT_proj, color='black', 
#         s=350, marker='s')

# plt_labels = hv.Labels(
#     {'x': chunk_centers.geometry.x.values, 
#     'y': chunk_centers.geometry.y.values, 
#     'text': chunk_centers.Site.values}, 
#     ['x','y'], 'text').opts(
#         yoffset=20000, color='black', 
#         size=28)



# elev_plt = hv.Image(xr_DEM.values, bounds=tpl_bnds).opts(
#     cmap='dimgray', colorbar=False)

# # Generate contour plot
# cont_plt = hv.operation.contours(elev_plt, levels=15).opts(
#     cmap='cividis', show_legend=False, 
#     colorbar=True, linewidth=2)

# # Generate elevation hillshade
# xr_HS = hillshade(xr_DEM)
# hill_plt = hv.Image(xr_HS.values, bounds=tpl_bnds).opts(
#         alpha=0.25, cmap='gray', colorbar=False)

# # Residuals plot
# plt_res = gv.Points(
#     data=gdf_PAIPR, crs=ANT_proj, vdims=['accum_res']).opts(
#         projection=ANT_proj, color='accum_res', s=50,
#         colorbar=True, cmap='coolwarm_r', symmetric=True)
# plt_res = plt_res.redim.range(accum_res=(res_min,res_max))




# data_map = (
#     hill_plt
#     * cont_plt.opts(colorbar=False)
#     * plt_locCOMB * plt_manPTS 
#     * (plt_accum2011 * plt_accum2016) 
#     * plt_labels.opts(size=36)
#     ).opts(fontscale=2, aspect=1, fig_inches=12)

# res_map = (
#     hill_plt
#     * cont_plt.opts(colorbar=False)
#     * (plt_manPTS*plt_res)
#     * plt_labels.opts(size=36)
#     ).opts(
#         fontscale=1.5, fig_inches=12)














# %%[markdown]

# ## Other misc snippets of old code


# %%

# fig_list = [col1, col2, col3, col4]
# ts_fig = plt.figure()
# # ts_fig = plt.subplots(
# #     ncols=len(fig_list), nrows=len(chunk_centers))

# for i, fig in enumerate(fig_list):
#     axes = fig.axes
#     for j, ax in enumerate(axes):

#         ax.remove()

#         ax.figure=ts_fig
#         ts_fig.axes.append(ax)
#         ts_fig.add_axes(ax)

#         dummy = ts_fig.add_subplot(j+1,i+1,1)
#         ax.set_position(dummy.get_position())
#         dummy.remove()

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

# from bokeh.io import export_svgs

# def export_svg(obj, filename):
#     plot_state = hv.renderer('bokeh').get_plot(obj).state
#     plot_state.output_backend = 'svg'
#     export_svgs(plot_state, filename=filename)

# print('svg plotting')
# export_svg(data_map, ROOT_DIR.joinpath(
#     'docs/Figures/oib-repeat/data_map.svg'))
# print('done with first plot')
# export_svg(fig_supp2, ROOT_DIR.joinpath(
#     'docs/Figures/oib-repeat/one2one_plts.svg'))
# print('done with second plot')




# ts_panel = pn.Row(
#     tsfig_PAIPR, tsfig_manual, tsfig_2011, tsfig_2016,
#     background='white')
# ts_panel.save(ROOT_DIR.joinpath(
#     'docs/Figures/oib-repeat/ts_ALL.png'))

# %%
