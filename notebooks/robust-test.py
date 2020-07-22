#%% [markdown]

# # Repeatability tests
# 
# This notebook serves as a test of the repeatability of results between different flightlines.
# To do this, I use the flightlines from 2011-11-09 in conjunction with a largely repeat flightline over the same area from 2016-11-09.
# I include traces that intersect between the flightlines with 50 m of one another.
# I further subset the data to a consistent time frame of 1990-2009.
# At various points in this analysis I compare annual accumulation estimates, mean trace accumulation estimates, and linear trends between the two flightlines (robust trend analysis performed using bootstrap resampling).

# *Note: You can ignore the code present throughout (unless you are interested in how I am doing things) and just focus on notes and figures.
# 
#%%
# Import requisite modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path

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
    data_0, start_yr=1996, end_yr=2005).drop(
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
    data_0, start_yr=1996, end_yr=2005).drop(
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

groups_2011 = data_2011.groupby(
    ['trace_ID', 'collect_time'])
traces_2011 = groups_2011.mean()[
    ['Lat', 'Lon']]
# traces_2011['idx'] = traces_2011.index
traces_2011 = traces_2011.reset_index()
gdf_2011 = gpd.GeoDataFrame(
    traces_2011[['trace_ID', 'collect_time']], 
    geometry=gpd.points_from_xy(
    traces_2011.Lon, traces_2011.Lat), 
    crs="EPSG:4326")

groups_2016 = data_2016.groupby(
    ['trace_ID', 'collect_time'])
traces_2016 = groups_2016.mean()[
    ['Lat', 'Lon']]
# traces_2016['idx'] = traces_2016.index
traces_2016 = traces_2016.reset_index()
gdf_2016 = gpd.GeoDataFrame(
    traces_2016[['trace_ID', 'collect_time']], 
    geometry=gpd.points_from_xy(
    traces_2016.Lon, traces_2016.Lat), 
    crs="EPSG:4326")

#%%





#%%[markdown]



#%%
df_dist = nearest_neighbor(
    gdf_2011, gdf_2016, return_dist=True)
idx_2011 = df_dist['distance'] <= 50
dist_2016 = df_dist[idx_2011]


gdf_traces = gpd.GeoDataFrame(
    {'ID_2011': gdf_2011.trace_ID[idx_2011], 
    'ID_2016': dist_2016.trace_ID}, 
    geometry=gdf_2011.geometry[idx_2011]).reset_index()

# Convert trace crs to same as Antarctic outline
gdf_traces = gdf_traces.to_crs(ant_outline.crs)

# Create numpy arrays for relevant results
accum_2011 = a2011_ALL[
    a2011_ALL.columns[idx_2011]].to_numpy()
std_2011 = std2011_ALL[
    std2011_ALL.columns[idx_2011]].to_numpy()
accum_2016 = a2016_ALL.iloc[:,dist_2016.idx].to_numpy()
std_2016 = std2016_ALL.iloc[:,dist_2016.idx].to_numpy()

#%% 
# Calculate and compare robust linear regression
trend_2011, lb_2011, ub_2011 = trend_bs(
    pd.DataFrame(accum_2011, index=a2011_ALL.index), 
    500)

trend_2016, lb_2016, ub_2016 = trend_bs(
    pd.DataFrame(accum_2016, index=a2016_ALL.index), 
    500)

accum_res = accum_2016 - accum_2011
accum_mu = np.mean([accum_2011, accum_2016], axis=0)

gdf_traces['accum2011'] = accum_2011.mean(axis=0)
gdf_traces['accum2016'] = accum_2016.mean(axis=0)
gdf_traces['accum_res'] = accum_res.mean(axis=0) \
    / accum_mu.mean(axis=0)
gdf_traces['trend2011'] = (trend_2011 
    / accum_mu.mean(axis=0))
gdf_traces['trend2016'] = (trend_2016 
    / accum_mu.mean(axis=0))
gdf_traces['2011 lb'] = (lb_2011 
    / accum_mu.mean(axis=0)) 
gdf_traces['2011 ub'] = (ub_2011 
    / accum_mu.mean(axis=0)) 
gdf_traces['2016 lb'] = (lb_2016 
    / accum_mu.mean(axis=0)) 
gdf_traces['2016 ub'] = (ub_2016 
    / accum_mu.mean(axis=0))
gdf_traces['trend_res'] = (gdf_traces['trend2016'] 
    - gdf_traces['trend2011'])
# gdf_traces['trend_res'] = ((trend_2016 - trend_2011) 
#     / pd.concat(
#         [trend_2011,trend_2016], axis=1).mean(axis=1))

res_perc = accum_res / accum_mu

# Create dataframes for scatter plots
accum_df = pd.DataFrame(
    {'Year': np.reshape(np.repeat(
        a2011_ALL.index, accum_2011.shape[1]), 
        std_2011.size), 
    'accum_2011': \
        np.reshape(accum_2011, accum_2011.size), 
    'std_2011': np.reshape(std_2011, std_2011.size), 
    'accum_2016': \
        np.reshape(accum_2016, accum_2016.size), 
    'std_2016': np.reshape(std_2016, std_2016.size)})

# # Determine which traces are statistically the same
# trends = pd.DataFrame({
#     '2011 trend': trend_2011/accum_2011.mean(axis=0), 
#     '2016 trend': trend_2016/accum_2011.mean(axis=0), 
#     '2011 lb': lb_2011 / accum_2011.mean(axis=0), 
#     '2011 ub':  ub_2011 / accum_2011.mean(axis=0), 
#     '2016 lb':lb_2016 / accum_2011.mean(axis=0), 
#     '2016 ub': ub_2016 / accum_2011.mean(axis=0)})

#%%[markdown]
# The following are some summary statistics regarding the bias between results of the 2011 flight and the 2016 flight.
# %%
print(
    f"The mean bias in annual accumulation "
    f"between 2011 and 2016 flights is "
    f"{res_perc.mean()*100:.2f}% "
    f"with a RMSE of {res_perc.std()*100:.2f}% "
    f"and a RMSE in mean accumulation of {gdf_traces.accum_res.std()*100:.2f}%"
)

print(
    f"The RMSE values compare favorably to the mean "
    f"standard deviations of the annual accumulation "
    f"estimates of "
    f"{(std_2011/accum_2011).mean()*100:.2f}% for " f"2011 and "
    f"{(std_2016/accum_2016).mean()*100:.2f}% for 2016."
)


# tmp_res = gdf_traces.trend_res.mean()*100
# tmp_rmse = (gdf_traces.trend_res.std()*100 - tmp_res) / tmp_res
# print(
#     f"The mean annual trend bias between 2011 and 2016 flights is {tmp_res:.2f}% "
#     f"with a RMSE of {tmp_rmse:.2f}%"
# )

#%%
import geoviews as gv
import holoviews as hv
from cartopy import crs as ccrs
from bokeh.io import output_notebook
output_notebook()
hv.extension('bokeh')
gv.extension('bokeh')

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)

# Define Antarctic boundary file
shp = str(ROOT_DIR.joinpath('data/Ant_basemap/Coastline_medium_res_polygon.shp'))

# Create data subset (for map plotting efficiency)
res_subset = gdf_traces.sample(2500)

#%%[markdown]
# The below plot shows the location of the data relative to the greater Antarctic region.
# 
# %%
## Plot data inset map
Ant_bnds = gv.Shape.from_shapefile(shp, crs=ANT_proj).opts(
    projection=ANT_proj, width=500, height=500)
trace_plt = gv.Points(res_subset, crs=ANT_proj).opts(
    projection=ANT_proj, color='red')
Ant_bnds * trace_plt

#%%[markdown]
# The below plot shows a 1:1 scatter plot between the 2011 flight (x-axis) and 2016 flight.
# It's also broken out by year, so you can use the slider to see the results from different years.
#  
#%%
one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=accum_df, 
    kdims=['accum_2011', 'accum_2016'], 
    vdims=['Year']).groupby('Year')
one_to_one.opts(color='black') * scatt_yr.opts(xlim=(100,750), ylim=(100,750))

#%%[markdown]
# Similar to above, but looking only at mean accumulation over the full period 1990-2009.
#   
#%%
one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[150,550], 'y':[150,550]}))
scatt_accum = hv.Points(
    data=pd.DataFrame(gdf_traces), 
    kdims=['accum2011', 'accum2016'], 
    vdims=[]
)
one_to_one.opts(color='black') * scatt_accum

#%%[markdown]
# This shows the 1:1 plot between flightlines for linear trends over the period of interest.
# Trends are expressed as a % change relative the the 2011 mean accumulation for each trace.
#  
#%%
one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[-0.04,0.03], 'y':[-0.04,0.03]}))
scatt_trend = hv.Points(
    data=pd.DataFrame(gdf_traces), 
    kdims=['trend2011', 'trend2016'], 
    vdims=[]
)
one_to_one.opts(color='black') * scatt_trend

#%%[markdown]
# The final two plots show the spatial distribution in residuals between 2011 and 2016 results for mean accumulation and linear trend.
# Both are expressed as % change relative to the 2011 mean accumulation rates.
#  
#%%
# Plot spatial distribution of mean accumulations for 2011 and 2016 flights side-by-side
a_plt2011 = gv.Points(
    res_subset, 
    vdims=gv.Dimension('accum2011', range=(150,550)), 
    crs=ANT_proj).opts(projection=ANT_proj, color='accum2011', 
    cmap='viridis', colorbar=True, 
    tools=['hover'], width=600, height=400)
a_plt2016 = gv.Points(
    res_subset, 
    vdims=gv.Dimension('accum2016', range=(150,550)),
    crs=ANT_proj).opts(projection=ANT_proj, color='accum2016', 
    cmap='viridis', colorbar=True, 
    tools=['hover'], width=600, height=400)
a_plt2011 + a_plt2016
# %%
# Plot spatial distribution of mean accum residual
accum_plt = gv.Points(
    res_subset, 
    vdims=['accum_res', 'accum2011', 'accum2016'], 
    crs=ANT_proj).opts(projection=ANT_proj, color='accum_res', 
    cmap='coolwarm', symmetric=True, colorbar=True, 
    tools=['hover'], width=600, height=400)
accum_plt

#%%
t_plt2011 = gv.Points(
    res_subset, 
    vdims=['trend2011', '2011 lb', '2011 ub'], 
    crs=ANT_proj). opts(
        projection=ANT_proj, color='trend2011', 
        cmap='coolwarm', symmetric=True, colorbar=True, 
        tools=['hover'], width=600, height=400)
t_plt2016 = gv.Points(
    res_subset, 
    vdims=['trend2016', '2016 lb', '2016 ub'], 
    crs=ANT_proj). opts(
        projection=ANT_proj, color='trend2016', 
        cmap='coolwarm', symmetric=True, colorbar=True, 
        tools=['hover'], width=600, height=400)
t_plt2011 + t_plt2016
# %%
# Spatial distribution in trend residuals
trends_plt = gv.Points(
    res_subset, vdims=['trend_res'], 
    crs=ANT_proj). opts(
        projection=ANT_proj, color='trend_res', 
        cmap='coolwarm', symmetric=True, colorbar=True, 
        tools=['hover'], width=600, height=400)
trends_plt


# %%
