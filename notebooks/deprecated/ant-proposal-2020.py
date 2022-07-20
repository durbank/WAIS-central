# %%[markdown]
# 
#  
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

# Set project root directory
ROOT_DIR = Path(__file__).parent.parent

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

core_accum = samba_raw.iloc[3:,1:]
core_accum.index = samba_raw.iloc[3:,0]
core_accum.index.name = 'Year'
core_meta = samba_raw.iloc[0:3,1:]
core_meta.index = ['Lat', 'Lon', 'Elev']
core_meta.index.name = 'Attributes'
new_row = core_accum.notna().sum()
new_row.name = 'Duration'
core_meta = core_meta.append(new_row)


core_accum = core_accum.transpose()
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
# Thomas compilation
ant2k_accum = pd.read_excel(
    DATA_DIR.joinpath('Ant2k_RegionalComposites_Thomas 2017_Dec_v2.xlsx'), 
    sheet_name='Original data', 
    header=6, index_col=0).transpose()
ant2k_cood = pd.read_excel(
    DATA_DIR.joinpath('Ant2k_RegionalComposites_Thomas 2017_Dec_v2.xlsx'), 
    sheet_name='Data Sources', header=4)
ant2k_cood = ant2k_cood[
    pd.notna(ant2k_cood['Latitude'])]
ant2k_locs = gpd.GeoDataFrame(
    {'Site': ant2k_cood['Site Name']}, 
    geometry=gpd.points_from_xy(
        ant2k_cood['Longitude'], 
        ant2k_cood['Latitude']), crs='EPSG:4326')
ant2k_locs = ant2k_locs.to_crs('EPSG:3031')

# %%
## Import flightline features
# # Define list of flightline shapes
# flight_list = [
#     file for file in DATA_DIR.glob(
#         'flightlines-all/*.shp')]
# # Define low-res mask for Antarctica
# gdf_mask = gpd.read_file(
#     gpd.datasets.get_path("naturalearth_lowres"))
# gdf_mask = gdf_mask[gdf_mask.continent=='Antarctica']
# # Import flightlines to gdf
# flights = gpd.GeoDataFrame(pd.concat(
#     [gpd.read_file(i) for i in flight_list], 
#     ignore_index=True), crs="EPSG:4326").to_crs(
#         epsg=3031)
# # Subset flights to within Antarctica
# tmp = gpd.sjoin(
#     flights, gdf_mask.to_crs(epsg=3031), 
#     how='inner', op='intersects')
# # Final gdf of Antarctic flightlines
# ant_flights = gpd.GeoDataFrame(
#     geometry=tmp.geometry)
# # Save output for future use
# ant_flights.to_file(
#     DATA_DIR.joinpath('ant_flights.geojson'), 
#     driver='GeoJSON')

# Import previously-generated Antarctic flightlines
ant_flights = gpd.read_file(
    DATA_DIR.joinpath('ant_flights.geojson'))

# Define low-res mask for Antarctica
gdf_mask = gpd.read_file(
    gpd.datasets.get_path("naturalearth_lowres"))
gdf_mask = gdf_mask[gdf_mask.continent=='Antarctica']
gdf_mask = gdf_mask.to_crs(epsg=3031)

ant_flights = gpd.clip(ant_flights, gdf_mask)
# %%
# Map of core and flightline locations
Ant_bnds = gv.Shape.from_shapefile(
    str(ant_path), crs=ANT_proj).opts(
    projection=ANT_proj, width=2000, height=2000)
flight_plt = gv.Path(
    ant_flights, crs=ANT_proj).opts(
        projection=ANT_proj, 
        alpha=0.33, color='red', line_width=2.5)
cores_plt = gv.Points(
    data=core_locs[core_locs['Duration'] >= 20], 
    vdims=['Name'], crs=ANT_proj).opts(
        projection=ANT_proj, color='blue', size=10, 
        tools=['hover'])
ant2k_plt = gv.Points(
    data=ant2k_locs, vdims=['Site'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='blue', size=10, 
        tools=['hover'])

Ant_bnds * flight_plt * cores_plt * ant2k_plt

# %%[markdown]





# %%
data_list = [dir for dir in DATA_DIR.glob('gamma/*/')]
print(f"Removed {data_list.pop(2)} from list")
print(f"Removed {data_list.pop(-1)} from list")
print(f"Removed {data_list.pop(2)} from list")
data_raw = pd.DataFrame()
for dir in data_list:
    data = import_PAIPR(dir)
    data_raw = data_raw.append(data)

# Subset data into QC flags 0, 1, and 2
data_0 = data_raw[data_raw['QC_flag'] == 0]
data_1 = data_raw[data_raw['QC_flag'] == 1]
# data_2 = data_raw[data_raw['QC_flag'] == 2]

# Remove data_1 values earlier than assigned QC yr, 
# and recombine results with main data results
data_1 = data_1[data_1.Year >= data_1.QC_yr]
data_0 = data_0.append(data_1).sort_values(
    ['collect_time', 'Year'])

#%%
# Format accumulation data
accum_long = format_PAIPR(
    data_0, start_yr=1985, end_yr=2009).drop(
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

# # Add std as % error of mean accum
# accum_trace['a_ERR'] = (
#     accum_trace['std'] / accum_trace['accum'])

# %%
# Calculate robust linear trends for each trace
trends, t_lb, t_ub = trend_bs(accum, 500)

# Add scaled trend values gdf
accum_trace['trends'] = trends / accum_trace.accum
accum_trace['trnd_lb'] = t_lb / accum_trace.accum
accum_trace['trnd_ub'] = t_ub / accum_trace.accum
accum_trace['trnd_ERR'] = pd.DataFrame({
    'ERR1': (trends - t_lb)/accum_trace.accum, 
    'ERR2': (t_ub - trends)/accum_trace.accum}).mean(
        axis=1)
accum_trace['sig'] = ~np.logical_and(
    (t_lb <= 0), (t_ub >= 0))

# %%
# Define plotting subset (to aid rendering times)
xmin, ymin, xmax, ymax = accum_trace.total_bounds
accum_subset = (
    accum_trace.cx[-1.42E6:xmax, ymin:ymax]
    .sample(5000)).sort_index()

# Plot data inset map
Ant_bnds = gv.Shape.from_shapefile(
    str(ant_path), crs=ANT_proj).opts(
    projection=ANT_proj, width=500, height=500)
trace_plt = gv.Points(
    accum_subset, crs=ANT_proj).opts(
    projection=ANT_proj, color='red')
Ant_bnds * trace_plt

# %%
# Plot mean accumulation across study region
accum_plt = gv.Points(
    accum_subset,vdims=['accum'], 
    crs=ANT_proj).opts(projection=ANT_proj, color='accum', 
    cmap='viridis', colorbar=True, 
    tools=['hover'], 
    width=1200, height=800)
# std_plt = gv.Points(
#     accum_subset, vdims=['a_ERR'], 
#     crs=ANT_proj).opts(
#         projection=ANT_proj, color='a_ERR', 
#         cmap='plasma', colorbar=True, 
#         tools=['hover'], width=600, height=400)
# accum_plt + std_plt

# %%
trend_plt = gv.Points(
    data=accum_subset, 
    vdims=['trends', 'trnd_lb', 'trnd_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trends', 
        cmap='coolwarm_r', symmetric=True, colorbar=True, 
        tools=['hover'], width=1200, height=800)
tERR_plt = gv.Points(
    data=accum_subset, vdims=['trnd_ERR'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trnd_ERR', 
        cmap='plasma', colorbar=True, 
        tools=['hover'], width=600, height=400)
# trend_plt + tERR_plt
(
    accum_plt.opts(fontscale=2.5, size=7.5, 
        width=1000, height=1000) 
    + trend_plt.opts(fontscale=2.5, size=7.5, 
        width=1000, height=1000)
)
# %%[markdown]




# %%
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

# %%
# %%
## 2011 cross-over comparison 1

Xover_2011 = gpd.GeoDataFrame(geometry=
    [Point(-1303550, -132950), 
    Point(-1078100, -419200)], crs='EPSG:3031')

# Trace_IDs of cross-over traces
ts_trace1, ts_trace2 = get_Xovers(
    Xover_2011[0:1], gdf_2011, 50)

# Extract cross-over time series and generate plots
plt1_ts1, plt1_ts2 = plot_Xover(
    a2011_ALL, std2011_ALL, ts_trace1, ts_trace2)

# Calculate trend of cross-over time series
trends1, lb1, ub1 = trend_bs(
    a2011_ALL[[ts_trace1, ts_trace2]], 500)

# Create array comparing all crossover results (trace1 is always the most recent result)
Xover_all = pd.DataFrame({
    'trace1':a2011_ALL[ts_trace1], 
    'trace2': a2011_ALL[ts_trace2]})

# %%
## 2011 crossover comparison 2

# Trace_IDs of cross-over traces
ts_trace1, ts_trace2 = get_Xovers(
    Xover_2011[1:], gdf_2011, 75)

# Extract cross-over time series and generate plots
plt2_ts1, plt2_ts2 = plot_Xover(
    a2011_ALL, std2011_ALL, ts_trace1, ts_trace2)

# Calculate trend of cross-over time series
trends2, lb2, ub2 = trend_bs(
    a2011_ALL[[ts_trace1, ts_trace2]], 500)

# Append crossover comparisons to array
Xover_all = Xover_all.append(
    pd.DataFrame({
    'trace1':a2011_ALL[ts_trace1], 
    'trace2': a2011_ALL[ts_trace2]})
)

# %% 
## 2016 cross-over comparison

Xover_2016 = gpd.GeoDataFrame(geometry=
    [Point(-1015500, -463900)], crs='EPSG:3031')


ts_trace1, ts_trace2 = get_Xovers(
    Xover_2016, gdf_2016, 50)

plt_ts1, plt_ts2 = plot_Xover(
    a2016_ALL, std2016_ALL, ts_trace1, ts_trace2)

# Calculate trend of cross-over time series
trends3, lb3, ub3 = trend_bs(
    a2011_ALL[[ts_trace1, ts_trace2]], 500)

Xover_all = Xover_all.append(
    pd.DataFrame({
    'trace1':a2016_ALL[ts_trace1], 
    'trace2': a2016_ALL[ts_trace2]})
)

# %%
# 1:1 plot of crossover points
one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,500], 'y':[100,500]}))
Xover_accum = hv.Points(
    data=Xover_all, 
    kdims=['trace1', 'trace2'], vdims=[])

# %%
# Create df of trend results for trace1 and trace2
tmp_trnd = [
    trends1.iloc[0], trends2.iloc[0], trends3.iloc[0]]
t1_trnd = pd.DataFrame({
    'x-pos': np.arange(0,3),
    'trend': tmp_trnd, 
    'negERR': (tmp_trnd 
        - np.array([lb1[0], lb2[0], lb3[0]])), 
    'posERR': (np.array([ub1[0], ub2[0], ub3[0]])
        - tmp_trnd)})
tmp_trnd = [
    trends1.iloc[1], trends2.iloc[1], trends3.iloc[1]]
t2_trnd = pd.DataFrame({
    'x-pos': np.arange(0,3) + 0.2,
    'trend': tmp_trnd, 
    'negERR': (tmp_trnd 
        - np.array([lb1[1], lb2[1], lb3[1]])), 
    'posERR': (np.array([ub1[1], ub2[1], ub3[1]])
        - tmp_trnd)})

t1_plt = (
    hv.Points(t1_trnd).opts(color='blue', size=5) 
    * hv.ErrorBars(
        t1_trnd, 
        vdims=['trend', 'negERR', 'posERR']).opts(
            color='blue')
)
t2_plt = (
    hv.Points(t2_trnd).opts(color='red', size=5) 
    * hv.ErrorBars(
        t2_trnd, 
        vdims=['trend', 'negERR', 'posERR']).opts(
            color='red')
)

# %%

(
    (one_to_one.opts(color='black') 
        * Xover_accum.opts(
            xlim=(100,500), ylim=(100,500), 
            xlabel='Trace1 accum (mm/yr)', 
            ylabel='Trace2 accum (mm/yr)', 
            width=700, height=700))
    + (t1_plt * t2_plt).opts(
        xticks=[(0,'Xover1'), (1,'Xover2'), 
            (2,'Xover3')], 
        xlabel='', ylabel='trend (mm/yr)', 
        width=700, height=700, tools=['hover'])
)

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
accum_2016 = a2016_ALL.iloc[:,dist_2016.trace_ID].to_numpy()
std_2016 = std2016_ALL.iloc[:,dist_2016.trace_ID].to_numpy()

#%%
# Calculate robust linear trends for each year's data
trend_2011, lb_2011, ub_2011 = trend_bs(
    pd.DataFrame(accum_2011, index=a2011_ALL.index), 
    500)
trend_2016, lb_2016, ub_2016 = trend_bs(
    pd.DataFrame(accum_2016, index=a2016_ALL.index), 
    500)

# Calculate mean annual accum and accum residuals
accum_res = accum_2016 - accum_2011
accum_mu = np.mean([accum_2011, accum_2016], axis=0)

# Add accum residual results to gdf
gdf_traces['accum2011'] = accum_2011.mean(axis=0)
gdf_traces['std_2011'] = (std_2011/accum_2011).mean(
    axis=0)
gdf_traces['accum2016'] = accum_2016.mean(axis=0)
gdf_traces['std_2016'] = (std_2016/accum_2016).mean(
    axis=0)
gdf_traces['accum_res'] = accum_res.mean(axis=0) \
    / accum_mu.mean(axis=0)

# %%
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

res_perc = accum_res / accum_mu

#%%
# Add trend difference stats to gdf
gdf_traces['trend2011'] = (trend_2011) 
    #/ accum_mu.mean(axis=0))
gdf_traces['trend2016'] = (trend_2016) 
    #/ accum_mu.mean(axis=0))
gdf_traces['2011 lb'] = (lb_2011) 
    # / accum_mu.mean(axis=0)) 
gdf_traces['2011 ub'] = (ub_2011) 
    # / accum_mu.mean(axis=0)) 
gdf_traces['2016 lb'] = (lb_2016) 
    # / accum_mu.mean(axis=0)) 
gdf_traces['2016 ub'] = (ub_2016) 
    # / accum_mu.mean(axis=0))
gdf_traces['trend_res'] = (gdf_traces['trend2016'] 
    - gdf_traces['trend2011'])

#%%
trend2011_ERR = np.array(
    [(ub_2011 - trend_2011), 
    (trend_2011 - lb_2011)])
trend2016_ERR = np.array(
    [(ub_2016 - trend_2016), 
    (trend_2016 - lb_2016)])

gdf_traces['trend2011_ERR'] = trend2011_ERR.mean(axis=0)
gdf_traces['trend2016_ERR'] = trend2016_ERR.mean(axis=0)

# %%
## Plot data inset map
# Create data subset (for map plotting efficiency)
res_subset = gdf_traces.sample(2500)

trace_plt = gv.Points(res_subset, crs=ANT_proj).opts(
    projection=ANT_proj, color='red')
Ant_bnds * trace_plt

#%%
one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,600], 'y':[100,600]}))
scatt_accum = hv.Points(
    data=pd.DataFrame(gdf_traces), 
    kdims=['accum2011', 'accum2016'], vdims=[])
dist_2011 = hv.Distribution(
    data=scatt_accum, kdims=['accum2011'])
dist_2016 = hv.Distribution(
    data=scatt_accum, kdims=['accum2016'])
accum_dist = hv.Distribution(
    data=gdf_traces.accum_res)

(hv.Layout(
    (one_to_one.opts(color='black') 
    * scatt_accum.opts(
        xlim=(100,600), ylim=(100,600), 
        xlabel='accum2011', ylabel='accum2016')) 
    << dist_2016.opts(width=130, xlim=(100,600)) 
    << dist_2011.opts(height=130, xlim=(100,600)))
    + accum_dist)

#%%
# Plot spatial distribution of mean accum residual
accum_plt = gv.Points(
    res_subset, 
    vdims=['accum_res', 'accum2011', 'accum2016'], 
    crs=ANT_proj).opts(projection=ANT_proj, color='accum_res', 
    cmap='coolwarm', symmetric=True, colorbar=True, 
    clabel='% Bias', 
    tools=['hover'], width=750, height=500)
accum_plt.relabel('Mean accum residuals')

#%%
one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[-15,15], 'y':[-15,15]}))
scatt_trend = hv.Points(
    data=pd.DataFrame(gdf_traces), 
    kdims=['trend2011', 'trend2016'], 
    vdims=[])
dist_2011 = hv.Distribution(
    scatt_trend, kdims=['trend2011']).opts(xlim=(-15,15))
dist_2016 = hv.Distribution(
    scatt_trend, kdims=['trend2016']).opts(xlim=(-15,15))
trends_hist = hv.Distribution(
    gdf_traces.trend_res).opts(xlabel='Linear trend (mm/yr^2)')

(hv.Layout(
    (one_to_one.opts(color='black') 
    * scatt_trend.opts(xlim=(-15,15), ylim=(-15,15), 
    xlabel='trend2011', ylabel='trend2016')) 
    << dist_2016.opts(width=150) 
    << dist_2011.opts(height=150))
    + trends_hist.relabel('Dist in trend residuals'))

# %%
# Spatial distribution in trend residuals
trends_plt = gv.Points(
    res_subset, vdims=['trend_res'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trend_res', 
        cmap='coolwarm', symmetric=True, colorbar=True, clabel='mm/yr^2', 
        tools=['hover'], width=1400, height=800)

# %%
# Plot of spatial accum with density
res_plt1 = (
    accum_plt.opts(
        width=1500, height=800, size=10, fontscale=2.5) 
    + accum_dist.opts(
        xlabel='accum residual (% bias)', 
        ylabel='', width=800, height=800, fontscale=2.5)
)

# Plot of spatial trends with trend densities
res_plt2 = (
    trends_plt.opts(
        width=1500, height=800, size=10, fontscale=2.5)
    + trends_hist.opts(
        xlabel='trend residual (mm/yr^2)', 
        ylabel='', width=800, height=800, fontscale=2.5)
)

res_plts = (res_plt1 + res_plt2).cols(2)
gv.save(res_plts, '/home/durbank/Documents/scratch/figs/res_plts.png')

# %% [markdown]















# %%[markdown]









# %%
# # Plot linear temporal accumulation trends
# trends_insig = gv.Points(
#     accum_subset[~accum_subset.sig], 
#     vdims=['trnd_perc', 'trnd_lb', 'trnd_ub'], 
#     crs=ANT_proj). opts(
#         alpha=0.05, projection=ANT_proj, color='trnd_perc', 
#         cmap='coolwarm_r', symmetric=True, colorbar=True, 
#         tools=['hover'], width=600, height=400)
# trends_sig = gv.Points(
#     accum_subset[accum_subset.sig], 
#     vdims=['trnd_perc', 'trnd_lb', 'trnd_ub'], 
#     crs=ANT_proj). opts(
#         projection=ANT_proj, color='trnd_perc', 
#         cmap='coolwarm_r', symmetric=True, colorbar=True, 
#         tools=['hover'], width=600, height=400)
# trends_insig * trends_sig