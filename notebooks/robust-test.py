#%% [markdown]

# # Repeatability tests
# 
# This notebook serves as a test of the repeatability of results between different flightlines.
# To do this, we use the flightlines from 2011-11-09 in conjunction with a largely repeat flightline over the same area from 2016-11-09.

#%%
# Import requisite modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import time

# Set project root directory
ROOT_DIR = Path(__file__).parent.parent

# Set project data directory
DATA_DIR = ROOT_DIR.joinpath('data')

# Import custom project functions
import sys
SRC_DIR = ROOT_DIR.joinpath('src')
sys.path.append(str(SRC_DIR))
from my_functions import *

#%%
# Import PAIPR-generated data
dir1 = DATA_DIR.joinpath('gamma/20111109/')
data_0 = import_PAIPR(dir1)
data_0 = data_0[data_0['QC_flag'] == 0].drop(
    'QC_flag', axis=1)
data_2011 = format_PAIPR(
    data_0, start_yr=1990, end_yr=2009).drop(
    'elev', axis=1)
a2011_ALL = data_2011.pivot(
    index='Year', columns='trace_ID', values='accum')
std2011_ALL = data_2011.pivot(
    index='Year', columns='trace_ID', values='std')


dir2 = DATA_DIR.joinpath('gamma/20161109/')
data_0 = import_PAIPR(dir2)
data_0 = data_0[data_0['QC_flag'] == 0].drop(
    'QC_flag', axis=1)
data_2016 = format_PAIPR(
    data_0, start_yr=1990, end_yr=2009).drop(
    'elev', axis=1)
a2016_ALL = data_2016.pivot(
    index='Year', columns='trace_ID', values='accum')
std2016_ALL = data_2016.pivot(
    index='Year', columns='trace_ID', values='std')

# Import Antarctic outline shapefile
ant_path = ROOT_DIR.joinpath(
    'data/Ant_basemap/Coastline_medium_res_polygon.shp')
ant_outline = gpd.read_file(ant_path)

# %% [markdown]
# This next bit utilizes some code derived from that available [here](https://automating-gis-processes.github.io/site/notebooks/L3/nearest-neighbor-faster.html).
# This uses `scikit-learn` to perform a much more optimized nearest neighbor search.
#%%

groups_2011 = data_2011.groupby('trace_ID')
traces_2011 = groups_2011.mean()[
    ['Lat', 'Lon']]
traces_2011['idx'] = traces_2011.index
traces_2011 = traces_2011.reset_index()
gpd_2011 = gpd.GeoDataFrame(
    traces_2011[['idx', 'trace_ID']], 
    geometry=gpd.points_from_xy(
    traces_2011.Lon, traces_2011.Lat), 
    crs="EPSG:4326")

groups_2016 = data_2016.groupby('trace_ID')
traces_2016 = groups_2016.mean()[
    ['Lat', 'Lon']]
traces_2016['idx'] = traces_2016.index
traces_2016 = traces_2016.reset_index()
gpd_2016 = gpd.GeoDataFrame(
    traces_2016[['idx', 'trace_ID']], 
    geometry=gpd.points_from_xy(
    traces_2016.Lon, traces_2016.Lat), 
    crs="EPSG:4326")




#%%
from sklearn.neighbors import BallTree

def get_nearest(
    src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""

    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)


def nearest_neighbor(left_gdf, right_gdf, return_dist=False, planet_radius=6371000):
    """
    For each point in left_gdf, find closest point in right GeoDataFrame and return them.
    When return_dist=True, also returns the distance between the nearest points for each entry.
    The planet radius (in meters) defaults to Earth's radius. 

    NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
    """

    left_geom_col = left_gdf.geometry.name
    right_geom_col = right_gdf.geometry.name

    # Ensure that index in right gdf is formed of sequential numbers
    right = right_gdf.copy().reset_index(drop=True)

    # Parse coordinates from points and insert them into a numpy array as RADIANS
    left_radians = np.array(
        left_gdf[left_geom_col].apply(
        lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
    right_radians = np.array(
        right[right_geom_col].apply(
        lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

    # Find the nearest points
    # -----------------------
    # closest ==> index in right_gdf that corresponds to the closest point
    # dist ==> distance between the nearest neighbors (in meters)

    closest, dist = get_nearest(src_points=left_radians, candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        closest_points['distance'] = dist * planet_radius

    return closest_points

#%%
df_dist = nearest_neighbor(
    gpd_2011, gpd_2016, return_dist=True)
idx_2011 = df_dist['distance'] <= 50
dist_2016 = df_dist[idx_2011]


gdf_traces = gpd.GeoDataFrame(
    {'ID_2011': gpd_2011.trace_ID[idx_2011], 
    'ID_2016': dist_2016.trace_ID}, 
    geometry=gpd_2011.geometry[idx_2011])

# Convert trace crs to same as Antarctic outline
gdf_traces = gdf_traces.to_crs(ant_outline.crs)

accum_2011 = a2011_ALL[
    a2011_ALL.columns[idx_2011]].to_numpy()
std_2011 = std2011_ALL[
    std2011_ALL.columns[idx_2011]].to_numpy()
accum_2016 = a2016_ALL.iloc[:,dist_2016.idx].to_numpy()
std_2016 = a2016_ALL.iloc[:,dist_2016.idx].to_numpy()

# %%

accum_res = accum_2016 - accum_2011
gdf_traces['res_mu'] = accum_res.mean(axis=0)



import matplotlib.pyplot as plt
fig, ax = plt.subplots()
gdf_traces.plot(
    column='res_mu', cmap='coolwarm', ax=ax, 
    legend=True)


#%% 
# Calculate and compare robust linear regression





# %%
test_2011 = accum_2011[:,0:25]
test_2016 = accum_2016[:,0:25]
test_res = test_2016 - test_2011

# fig, ax = plt.subplots()
# ax.plot([150, 500], [150,500], color='black')
# ax.scatter(test_2011, test_2016)

fig, ax = plt.subplots()
ax.plot([150, 500], [150,500], color='black')
ax.scatter(
    accum_2011.mean(axis=0), accum_2016.mean(axis=0))


fig, ax = plt.subplots()
ax.plot([100, 600], [100,600], color='black')
ax.scatter(accum_2011, accum_2016)
# %%

import geoviews as gv
from cartopy import crs as ccrs
from bokeh.io import output_notebook
output_notebook()
gv.extension('bokeh')

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)

# Define Antarctic boundary file
shp = str(ROOT_DIR.joinpath('data/Ant_basemap/Coastline_medium_res_polygon.shp'))

# %%
# Plot mean accumulation across study region
accum_plt = gv.Points(accum_subset,vdims=['accum', 'std'], 
    crs=ANT_proj).opts(projection=ANT_proj, color='accum', 
    cmap='viridis', colorbar=True, 
    tools=['hover'], width=600, height=400)
accum_plt

# %%
# Plot linear temporal accumulation trends
trends_insig = gv.Points(
    accum_subset[~accum_subset.sig], 
    vdims=['trnd_perc', 'trnd_lb', 'trnd_ub'], 
    crs=ANT_proj). opts(
        alpha=0.05, projection=ANT_proj, color='trnd_perc', 
        cmap='coolwarm_r', symmetric=True, colorbar=True, 
        tools=['hover'], width=600, height=400)
trends_sig = gv.Points(
    accum_subset[accum_subset.sig], 
    vdims=['trnd_perc', 'trnd_lb', 'trnd_ub'], 
    crs=ANT_proj). opts(
        projection=ANT_proj, color='trnd_perc', 
        cmap='coolwarm_r', symmetric=True, colorbar=True, 
        tools=['hover'], width=600, height=400)
trends_insig * trends_sig