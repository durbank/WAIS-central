#%%[markdown]
# # Proto-paper on central WAIS accumulation patterns and trends
# 
# This serves as a (very) initial rough draft of the direction and thoughts regarding this next paper.
# It discusses the PAIPR results from 5 flightlines in central WAIS, including the spatial variability and patterns of temporal trends in SMB.
# I've included additional figures, notes, analysis, and thoughts that won't make it into the final paper, but hope these might help discuss things at a greater level of detail than would be possible otherwise.
# I'm thinking GRL could be a good target for this article, but open to all suggestions.
# The text portions are still quite rough, but I've tried to include the general structure I envision for the paper in bold, with thoughts and sentence fragments of what to include in various paragraphs.
# Still very much a work in progress. 
# 
# ## Introduction
# 
# **Paragraph on the importance of understanding recent accumulation in West Antarctica.**
# West Antarctica is highly relevant to sea level projections (recent citations of how much in is likely to contribute in the near future), particularly as recent evidence suggests as much as 24% of the ice sheet is in a state of dynamical inbalance (Shepherd et. al., 2019; likely other important citations I could include).
# The surface mass balance (SMB) is relevant for sea level rise calculations, both directly as SMB contributes to overall mass balance and indirectly as SMB affects estimates of ice sheet mass change from altimetry measurements (recent citations from IceSat, IceSat-2, and radar altimetry - Shepherd et. al., 2018; Rignot et. al., 2019; etc.).
# Additionally, understanding the recent spatiotemporal trends and patterns in accumulation over recent decades provides insight into how atmospheric and oceanic dynamics are changing in the Southern Hemisphere and globally (a couple of recent citations supporting this logic).
# 
# **Paragraph on the difficulties in acquiring accumulation estimates.**
# Despite the critical need for increased understanding, our knowledge of West Antarctic SMB is hampered by a paucity of data and data coverage.
# The extremely remote and harsh environment of the region make *in-situ* measurements challenging and expensive to collect.
# SMB is often highly spatially variable (Lenaerts et. al., 2019?), leading to questions of how representative the relatively small sample of point source measurements are of the ice sheet as a whole.
# Satellite-based estimates are not entirely straightforward, usually have coarse spatial resolution, and limited temporal extent.
# Modeling results have seen enormous strides forward, but validation with independent estimates of SMB are still important, especially in data-sparse regions like West Antarctica.
# Ground-based and airborne microwave radar has been used for decades (e.g. Spikes et. al., 2004; Anshultz et. al., 2007; MagGregor et al., 2009; Medley et. al., 2013; etc.), but still suffers from limitations of access and processing bottlenecks usually involving labor-intensive manual work.
# 
# **Paragraph summarizing recent research investigating accumulation in West Antarctica**
# Despite these limitations, significant progress has been made towards addressing these questions.
# Dattler et. al., 2019 for recent radar insights (probably need to find others as well).
# Summary of most recent ice core findings (Thomas et. al., 2017; Medley and Thomas, 2019; Wang et. al., 2016, likely others?).
# Summary of recent modeling work (Lenaerts et. al., 2018; Lenaerts et. al., 2019).
# 
# **Paragraph explicitly stating some of the unknowns in all of this (leading to next paragraph of how we will answer some of these).**
# None of these, however, have investigated trends in SMB over recent decades from radar (permitting the spatial resolution and coverage desired).
# Despite these advancements, important questions remain regarding whether the SMB of West Antarctica is increasing or decreasing, along with important questions of positive, negative, and stable regions and the boundaries between these.
# Peninsula appears to be increasing (Thomas papers).
# Ross region has evidence of decreased accumulation over recent decades (Kaspari et. al., 2004; Wang et. al., 2017).
# Some studies show relatively stable SMB over recent decades (Medley 2013; Wang et. al., 2017).
# *{Should include some modeling papers as well and discuss if they agree with this or not.}*
# Discussion of observed dipole in accumulation in other work (Bertler et. al., 2018), suggestion of link to ASL (note that many details regarding this mechanism and its influence are not yet understood).
# Include any conflicting papers for these general trends.
# The precise location and nature of these transitions are essential to properly understanding the climatic elements driving the observed changes.
# 
# **Paragraph describing our research and its contribution to these questions.**
# We use a fully automated and self-QCing algorithm to investigate spatiotemporal patterns and trends in annually-resolved SMB across central WAIS.
# Include other summary details of methods, which questions these results address (and how they can address them), and reiterate why such results are noteworthy, significant, and important.
# 
# ## Materials and Methods
# 
# ### Data and Study Site
# 
# **Paragraph describing the study site and how it helps address questions.**
# This study focuses on a region of central West Antarctica covering {give region size}.
# It lies between the Antarctic Peninsula and Ellsworth Land (where prior studies suggest increased SMB in recent decades) and the Ross Sea and Marie Byrd Land (where evidence suggests a decrease in SMB in recent decades).
# The region further feeds important outlet glaciers such as Thwaites Glacier and Pine Island Glacier, which currently...(explain (with citations) recent impact of these glaciers and why they are important moving forward).
#  
# %% Set the environment

# Import modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
import geoviews as gv
import holoviews as hv
from cartopy import crs as ccrs
from bokeh.io import output_notebook
output_notebook()
hv.extension('bokeh')
gv.extension('bokeh')
from shapely.geometry import Polygon, Point

# Set project root directory
ROOT_DIR = Path(__file__).parents[2]

# Set project data directory
DATA_DIR = ROOT_DIR.joinpath('data/PAIPR-outputs')

# Import custom project functions
import sys
SRC_DIR = ROOT_DIR.joinpath('src')
sys.path.append(str(SRC_DIR))
from my_functions import *

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(
    true_scale_latitude=-71)

# Define Antarctic boundary file
shp = str(ROOT_DIR.joinpath(
    'data/Ant_basemap/Coastline_medium_res_polygon.shp'))
gdf_ANT = gpd.read_file(shp)

#%% Import and format PAIPR-generated results

# Import raw data
data_list = [folder for folder in DATA_DIR.glob('*')]
data_list = [
    path for path in data_list 
    if "20141103" not in str(path)] #Removes 2014 results, as there are too few and are suspect
data_raw = pd.DataFrame()
for folder in data_list:
    data = import_PAIPR(folder)
    data_raw = data_raw.append(data)

# Remove results for below QC data reliability
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)

# Format and sort results for further processing
data_form = format_PAIPR(data_0).drop(
    'elev', axis=1)

# Create time series arrays for annual accumulation 
# and error
accum_ALL = data_form.pivot(
    index='Year', columns='trace_ID', values='accum')
std_ALL = data_form.pivot(
    index='Year', columns='trace_ID', values='std')

# Create gdf of mean results for each trace and 
# transform to Antarctic Polar Stereographic
gdf_traces = long2gdf(data_form)
gdf_traces.to_crs(epsg=3031, inplace=True)

# %% Import and format SAMBA cores

# Import raw data
samba_raw = pd.read_excel(
    ROOT_DIR.joinpath("data/DGK_SMB_compilation.xlsx"), 
    sheet_name='Accumulation')

# Format SAMBA core data
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
# core_ALL = core_ALL.transpose()
core_meta = core_meta.transpose()
core_meta.index.name = 'Name'

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
# bbox = Polygon(
#     [[-1.684E6,9.102E4], [-1.684E6,-7.775E5],
#     [-4.709E5,-7.775E5], [-4.709E5,9.102E4]])

# Subset core results to region of interest
keep_idx = core_locs.within(bbox)
gdf_cores = core_locs[keep_idx]
core_ACCUM = core_ALL.loc[:,keep_idx].sort_index()

# %% Study site and data figure

# Radar data plot
radar_plt = gv.Points(
    gdf_traces.sample(2500), crs=ANT_proj).opts(
        projection=ANT_proj, color='red')

# Core data plot
core_plt = gv.Points(
    gdf_cores, crs=ANT_proj, 
    vdims=['Name']).opts(
        projection=ANT_proj, color='blue', size=5, 
        tools=['hover'])

# Antarctica boundaries
Ant_bnds = gv.Shape.from_shapefile(
    shp, crs=ANT_proj).opts(
        projection=ANT_proj, color='grey', 
        alpha=0.5)

# Add plot to workspace
(Ant_bnds * radar_plt * core_plt).opts(
    width=500, height=500)

# %%[markdown]
# **Paragraph describing the radar data used in the study.**
# This study uses 5 flightlines from NASA's Operation IceBridge Snow radar.
# This radar system consists of...{describe radar system, with citations of Rodriquez-Morales et. al., 2014}
# The radar used in this study represent over {give flightline distance} km of collected radar echograms.
# 
# **Paragraph describing and citing the ice cores used in the study.**
# We further use a collection of 41 ice and firn cores in central WAIS to compare to the radar-derived results.
# These records come from various different campaigns...
# Both datasets (OIB Snow radar and ice cores) are shown in Figure X.
# 
# ### Depth-density modeling
# 
# **Paragraph on how we generate the density estimates (probably ask Phil to write this).**
# 
# ### SMB estimation with PAIPR
# 
# **A couple of paragraphs desribing the PAIPR method.**
# This study utilizes the Probabalistic Annual Isochrone Picking Routine (PAIPR) to generate SMB time series (and associated uncertainty) from radar echograms.
# This is a fully-automated computer vision technique that utilizes Radon transforms and Monte Carlo sampling to ...{further explain about the method}.
# Further details of the method its validation in central WAIS are in Keeler et. al., 2020.
# 
# Since the original publication, PAIPR has received a number of improvements.
# Most significant is a quality control subroutine that assesses the reliability of PAIPR results for a given echogram. This allows for elimination of unreliable data without the need for manual oversight, greatly improving the usability of PAIPR.
# {Discuss the automated QC method and how it works, including removing data with overly large uncertainties of +/-2 sigma that intersects zero}.
# {Include additional summaries of any other updates since publication}.
# 
# **Paragraph describing the values of this approach.**
# {Also include important limitations or issues, and how we mitigate their effects}. 
# PAIPR generated SMB time series estimates at 25 m along-track resolution which are eventually aggregated to 250 m resolution.
# A total of 18,196 individual time series at this spacing were produced for this analysis. 
# 
# ### Data Processing and Trend Analysis
# 
# PAIPR-generated results first go through a number of preprocessing steps prior to analysis.
# The multiple flightlines making up the data result in overlapping results.
# We therefore aggregate all results within a given grid cell into a single time series, with a grid cell size of 2.5 km.
# We further subset the resultant dataset to time series with full coverage for 1980-2010 to ensure a consistent time period.
# After processing and filtering, the final dataset consists of 839 unique annually resolved time series of SMB (with annual uncertainty) covering a consistent 31-year time period.
# 
# %%

# Combine trace time series based on grid cells
tmp_grids = pts2grid(gdf_traces, resolution=2500)
(gdf_grid_ALL, accum_grid_ALL, 
    std_grid_ALL, yr_count_ALL) = trace_combine(
    tmp_grids, accum_ALL, std_ALL)

# Subset results to period 1980-2010
yr_start = 1980
yr_end = 2009
keep_idx = np.invert(
    accum_grid_ALL.loc[yr_start:yr_end,:].isnull().any())
accum_grid = accum_grid_ALL.loc[yr_start:yr_end,keep_idx]
std_grid = std_grid_ALL.loc[yr_start:yr_end,keep_idx]
gdf_grid = gdf_grid_ALL[keep_idx]
gdf_grid['accum'] = accum_grid.mean()
gdf_grid['std'] = np.sqrt((std_grid**2).mean())

# Subset cores to same time period
keep_idx = np.invert(
    core_ACCUM.loc[yr_start:yr_end,:].isnull().any())
accum_core = core_ACCUM.loc[yr_start:yr_end,keep_idx]
gdf_core = gdf_cores[keep_idx]
gdf_core['accum'] = accum_core.mean()

# %%[markdown]
# We generate estimates of the linear trend in annual SMB for each of the 924 time series, weighted by the uncertainty for each year.
# The high variability in annual SMB, combined with the short duration of the records, means these trend analyses can be unstable and are susceptible to leverage effects.
# To enhance the robustness of our results, we perform probabalistic bootstrapping with replacement (1000 simulations per time series) to minimize the influence of these issues and to generate 95% confidence intervals for the trend estimates. 
# 
# {Note somewhere that we perform the same filtering and trend analysis on our ice core results as we do on the radar data}. 
# 
# %%

# Calculate trends in radar
trends, _, lb, ub = trend_bs(
    accum_grid, 1000, df_err=std_grid)
gdf_grid['trend'] = trends / gdf_grid['accum']
gdf_grid['t_lb'] = lb / gdf_grid['accum']
gdf_grid['t_ub'] = ub / gdf_grid['accum']
gdf_grid['t_abs'] = trends

# Add factor for trend signficance
insig_idx = gdf_grid.query(
    't_lb<0 & t_ub>0').index.values
sig_idx = np.invert(np.array(
    [(gdf_grid['t_lb'] < 0).values, 
    (gdf_grid['t_ub'] > 0).values]).all(axis=0))

# Calculate trends in cores
trends, _, lb, ub = trend_bs(accum_core, 1000)
gdf_core['trend'] = trends / gdf_core['accum']
gdf_core['t_lb'] = lb / gdf_core['accum']
gdf_core['t_ub'] = ub / gdf_core['accum']
gdf_core['t_abs'] = trends

# Determine trend significance for cores
insig_core = gdf_core.query(
    't_lb<0 & t_ub>0').index.values
sig_core = np.invert(np.array(
    [(gdf_core['t_lb'] < 0).values, 
    (gdf_core['t_ub'] > 0).values]).all(axis=0))

# %%[markdown]
# ## Results
# 
# Figure X shows the mean accumulation (mm w.e. per year) generated from radar echograms using PAIPR (1980-2010) and the mean accumulation of the selected ice cores for the same time period (triangles).
# *NOTE: the following plots are dynamic, so it is possible to zoom and pan, with additional info available as a mouse-over. The figures are zoomed to the extent of the radar data by default, but there are additional cores included beyond this scope that can be viewed by zooming out or panning.* 
# 
#%% Plot of mean accumulation

ac_max = gdf_grid.accum.max()
ac_min = gdf_grid.accum.min()
gdf_bounds = {
    'x_range': tuple(gdf_grid.total_bounds[0::2]), 
    'y_range': tuple(gdf_grid.total_bounds[1::2])}
# ac_max = np.max(
#     [gdf_core.accum.max(), gdf_grid.accum.max()])
# ac_min = np.min(
#     [gdf_core.accum.min(), gdf_grid.accum.min()])

accum_plt = gv.Polygons(
    gdf_grid, 
    vdims=['accum', 'std'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='viridis', colorbar=True, 
        tools=['hover'])
accum_core_plt = gv.Points(
    gdf_core, vdims=['accum', 'Name'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, 
        size=7, marker='triangle', tools=['hover'])
count_plt = gv.Polygons(
    gdf_grid, vdims='trace_count', 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='magma', colorbar=True, 
        tools=['hover'])
(accum_plt*accum_core_plt).opts(
    width=600, height=400, 
    xlim=gdf_bounds['x_range'], 
    ylim=gdf_bounds['y_range']).redim.range(
    accum=(ac_min,ac_max))
# ((accum_plt*accum_core_plt).opts(
#     width=600, height=400).redim.range(accum=(ac_min,ac_max))
#     + count_plt.opts(width=600, height=400))

# %%[markdown]
# The results show high variability in mean accumulation, but the overall spatial patterns matche with previous observations of decreased SMB towards the interior of the ice sheet.
# The high spatial variability seen in the data also support previous findings, with the most likely cause topographic variation (Dattler et. al., 2019).
# 
# The linear trends are presented below in Figure X. 
# The left panel shows the linear trends (1980-2010) as % change per year relative to the multidecadal mean SMB.
# Insignificant trends are shown in grey, with ice cores displayed as triangles.
# The right panel shows the 95% confidence margin of error for the radar results, again expressed as % change per year relative to the multidecadal mean SMB.
# *NOTE: Similar to the previous plot, this plot defaults to the extent of the radar data, but additional core results are viewable beyond this extent.*
#  
#%% Plot linear trend results

t_max = np.max(
    [gdf_grid.trend.max(), gdf_core.trend.max()])
t_min = np.min(
    [gdf_grid.trend.min(), gdf_core.trend.min()])

gdf_ERR = gdf_grid.copy().drop(
    ['accum', 'std', 't_lb', 
    't_ub', 't_abs'], axis=1)
gdf_ERR['MoE'] = (
    gdf_grid['t_ub'] 
    - gdf_grid['t_lb']) / 2

sig_plt = gv.Polygons(
    gdf_grid[sig_idx], 
    vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, tools=['hover'])
insig_plt = gv.Polygons(
    gdf_grid.loc[insig_idx], 
    vdims=['trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color_index=None, 
        fill_alpha=0.75, color='grey', 
        line_color=None, tools=['hover'])
sig_core_plt = gv.Points(
    gdf_core[sig_core], 
    vdims=['Name', 'trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, size=7, line_color='black', 
        marker='triangle', tools=['hover'])
insig_core_plt = gv.Points(
    gdf_core.loc[insig_core], 
    vdims=['Name', 'trend', 't_lb', 't_ub'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='grey', alpha=0.75,  
        size=7, line_color='black', marker='triangle', tools=['hover'])
tERR_plt = gv.Polygons(
    gdf_ERR, vdims=['MoE', 'trend'], 
    crs=ANT_proj).opts(projection=ANT_proj, 
    line_color=None, cmap='plasma', colorbar=True, 
    tools=['hover'])
insig_plt*sig_plt*insig_core_plt*sig_core_plt.opts(
    width=600, height=400, 
    xlim=gdf_bounds['x_range'], 
    ylim=gdf_bounds['y_range']).redim.range(trend=(t_min,t_max)) + tERR_plt.opts(width=600, height=400, 
    xlim=gdf_bounds['x_range'], 
    ylim=gdf_bounds['y_range'])

# %%[markdown]
# The next figure provides additional context from a larger suite of surrounding ice cores.
# To achieve this, we include any nearby cores with at least some data in the period 1975-2016 and perform bootstrapped trend analysis on the time series.
# The following figure is therefore similar to the preceding figure, but with more surrounding cores.
# The size of the cores (triangles in the plot) represent the length of time used to calculate their trend, with the largest covering the full period (1975-2016) and the smallest covering only a handful of years.
# Ice cores tend to show increasing accumulation near the Peninsula, grading to insignificant and negative accumulation rates moving westward (down in the plot).
#  
# %% Trends with all cores (not all cover full time period)

# Keep all core data 1975 to present
cores_long = core_ACCUM.loc[1975:]
gdf_long = gdf_cores
gdf_long['accum'] = cores_long.mean()

# Create size variable based on number of years in record
tmp = np.invert(
    cores_long.isna()).sum()
gdf_long['size'] = 10*tmp/tmp.max()

# Calculate trends in cores
trends, _, lb, ub = trend_bs(cores_long, 1000)
gdf_long['trend'] = trends / gdf_long['accum']
gdf_long['t_lb'] = lb / gdf_long['accum']
gdf_long['t_ub'] = ub / gdf_long['accum']
gdf_long['t_abs'] = trends

# Determine trend significance for cores
insig_core = gdf_long.query(
    't_lb<0 & t_ub>0').index.values
sig_core = np.invert(np.array(
    [(gdf_long['t_lb'] < 0).values, 
    (gdf_long['t_ub'] > 0).values]).all(axis=0))

# Plot trends since 1975 (all)
t_max = np.max(
    [gdf_grid.trend.max(), gdf_long.trend.max()])
t_min = np.min(
    [gdf_grid.trend.min(), gdf_long.trend.min()])

sig_core_plt = gv.Points(
    gdf_long[sig_core], 
    vdims=['Name', 'trend', 't_lb', 't_ub', 'size'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm_r', symmetric=True, 
        colorbar=True, size='size', 
        line_color='black', 
        marker='triangle', tools=['hover'])
insig_core_plt = gv.Points(
    gdf_long.loc[insig_core], 
    vdims=['Name', 'trend', 't_lb', 't_ub', 'size'], 
    crs=ANT_proj).opts(
        projection=ANT_proj, color='grey', 
        alpha=0.75, size='size', line_color='black', 
        marker='triangle', tools=['hover'])
insig_plt*sig_plt*insig_core_plt*sig_core_plt.opts(
    width=600, height=400).redim.range(
        trend=(t_min,t_max))



# %%[markdown]
# In order to better investigate the temporal characteristics of annual SMB in the data, we aggregate our results into 100 km grid cells and generate mean time series for each grid cell.
# These grid cell time series are then grouped into 5 distinct regions that demonstrate similar temporal responses.
# The following plots show the time series within each group, along with companion plots showing the number of observations represented for each year in the 100-km grid composite records.
# Any cores within each group are also combined, with the composite core plotted in the figure as a dashed grey line.
# The final figure shows how the 100 km grid cells are divided into their respective groups.
# 
# %% Larger grid cells and composite time series

gdf_cores['accum'] = core_ACCUM.loc[
    1960::,gdf_cores.index].mean()

# Combine trace time series based on grid cells
tmp_grid_big = pts2grid(gdf_traces, resolution=100000)
(gdf_BIG, accum_BIG, 
    std_BIG, yr_count_BIG) = trace_combine(
    tmp_grid_big, accum_ALL, std_ALL)

# %%
PIG_group = [27, 28, 29, 34, 35]
DIV_group = [25, 26, 31, 32]
ROSS_group = [43, 49]
group_3 = [33, 37, 38]
group_4 = [39, 40]

grid_groups = [
    PIG_group, DIV_group, 
    ROSS_group, group_3, group_4]

import string
alphabet = list(string.ascii_uppercase)
poly_groups = []
group_cmap = {
    'A': 'midnightblue', 'B': 'mediumseagreen', 
    'C': 'purple', 'D': 'olive', 'E': 'saddlebrown'}

for i, group in enumerate(grid_groups):
    
    group_idx = [
        idx for idx, row in gdf_BIG.iterrows() 
        if row['grid_ID'] in group]
    gdf_group = gdf_BIG.loc[group_idx]
    accum_group = accum_BIG.loc[:,group_idx]
    count_group = yr_count_BIG.loc[:,group_idx]
    
    
    poly = gdf_group.geometry.unary_union
    cores_group = gdf_cores[
        gdf_cores.geometry.within(poly)]
    core_comp = core_ACCUM[
        cores_group.index].loc[
            accum_group.index[0]:].mean(axis=1)
    poly_groups.append(poly)

    
    

    
    fig, (ax1, ax2) = plt.subplots(2, 1)
    fig.suptitle(
        'Grid Group '+alphabet[i]+' time series', 
        color=group_cmap[alphabet[i]])
    accum_group.plot(ax=ax1, label='_hidden_')
    core_comp.plot(
        ax=ax1, color='grey', linestyle='--')
    count_group.plot(ax=ax2)
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    ax1.set_ylabel('SMB (mm/a)')
    ax2.set_ylabel('No. records')
    plt.show()


gdf_groups = gpd.GeoDataFrame(
    {'Group_ID': alphabet[0:len(poly_groups)]}, 
    geometry=poly_groups, crs=gdf_BIG.crs)


# a_max = gdf_BIG.accum.max()
# a_min = gdf_BIG.accum.min()
# # Big grid cell plot
# grid_plt = gv.Polygons(
#     gdf_BIG, vdims=['accum', 'trace_count', 'grid_ID'], 
#     crs=ANT_proj).opts(
#         projection=ANT_proj, color='accum',
#         line_color='black', tools=['hover'], 
#         colorbar=True, cmap='viridis')
# # Core data plot
# core_plt = gv.Points(
#     gdf_cores, crs=ANT_proj, 
#     vdims=['accum', 'Name']).opts(
#         projection=ANT_proj, color='accum', 
#         line_color='black', marker='triangle', 
#         cmap='viridis', size=10, tools=['hover'])
# (grid_plt * core_plt).opts(
#     width=700, height=450).redim.range(accum=(a_min,a_max))


# %%

bounds = {
    'x_range': tuple(gdf_BIG.total_bounds[0::2]), 
    'y_range': tuple(gdf_BIG.total_bounds[1::2])}

# Grid groups
group_plt = gv.Polygons(
    gdf_groups, vdims='Group_ID', 
    crs=ANT_proj).opts(
        projection=ANT_proj, line_color=None, 
        color_index='Group_ID', cmap=group_cmap, tools=['hover'], 
        fill_alpha=0.5)

# Big grid cell plot
grid_plt = gv.Polygons(
    gdf_BIG, 
    crs=ANT_proj).opts(
        projection=ANT_proj, 
        line_color='black', color_index=None, 
        color=None)

# Radar data plot
radar_plt = gv.Polygons(
    gdf_grid_ALL, crs=ANT_proj).opts(
        projection=ANT_proj, 
        line_color=None, color='red')

# Core data plot
core_plt = gv.Points(
    gdf_cores, crs=ANT_proj, 
    vdims=['Name']).opts(
        projection=ANT_proj, color='blue', 
        line_color='black', marker='triangle', 
        size=7, tools=['hover'])

# # Antarctica boundaries
# Ant_bnds = gv.Polygons(
#     gdf_ANT, crs=ANT_proj).opts(
#         projection=ANT_proj, color='grey', 
#         alpha=0.3)

# Add plot to workspace
(group_plt * grid_plt * radar_plt * core_plt).opts(
    width=700, height=450, xlim=bounds['x_range'], 
    ylim=bounds['y_range'])




# %%[markdown]
# ## Discussion
# 
# {Still very much a work in progress. Need to fit my observations and results into the greater context of recent literature}.
# Major points to discuss further:
# 
# - High variability in trend results
# - Although the majority of results show no significant trends, 35% show significant decreases in accumulation with a mean -1.14% per year over the period 1979-2010
# - The negative results are more prevalent in the western portion of the region, but are highly variable
# - Several previous studies suggested widespread stability in SMB rates for this region, although these were based off a small number of ice core records (Kaspari et. al., 2004; Wang et. al., 2017; Banta et. al., 2008; *need to check other more recent referecnes*) or else confined to a smaller spatial extent (Medley et. al., 2013)
# - Results supports previous observation of negative SMB in recent decades in a part of this region as reported by Burgner et. al. (2012) 
# 
# Important takeaways:
# 
# - The picture of recent SMB trends is more complicated than the typical cartoon picture of the Amundsen-Ross dipole in moisture from the ASL
# - Indicates significant portions of central WAIS are experiencing negative SMB trends, agreeing with Burgner et. al. (2012) as opposed to other reports of widespread stability
# - Shows negative trends significantly farther east than previous findings (even the PIG core, although in a local region of stability, shows negative SMB in surrounding areas)
# 
# Important caveats:
# 
# - Relatively short duration (~30 years), so questions of how much of the observed trends result from internal climate variabliity rather a distinct climatic change
# - Use of linear trends could obscure the onset of decline in SMB rates
#  
# %%[markdown]
# ## Supplement
# 
# This is all information that I think could be valulable in a supplement or appendix.
# Most of it has to do with further info on the methods, along with a more thorough validation of the method based on repeatability and comparisons to manually-traced layers.
# 
# ### Comparisons of radar and core time series
# 
# The following plots compare the radar results from before to nearby cores.
# This only looks at cores that cover the full 1979-2010 time period and that are within 10 km of a radar grid cell.
#  
#%% Compare core and radar time series

# Get radar grid centroids
grid_pts = gdf_grid.copy()
grid_pts['geometry'] = grid_pts['geometry'].centroid

df_dist = nearest_neighbor(
    gdf_core, grid_pts, return_dist=True)
idx_match = df_dist['distance'] <= 10000
dist_overlap = df_dist[idx_match]

# Create numpy arrays for relevant results
core_overlap = accum_core.iloc[
    :,dist_overlap.index]
radar_idx = []
for val in dist_overlap['grid_ID'].values:
    row = gdf_grid[gdf_grid['grid_ID'] == val]
    radar_idx.append(row.index)
r_idx = [val[0] for val in radar_idx]
accum_overlap = accum_grid.loc[:,r_idx]
std_overlap = std_grid.loc[:,r_idx]
gdf_overlap = gdf_grid.loc[r_idx]

for _ in range(len(gdf_overlap)):
    
    ts_core = core_overlap.iloc[:,_]
    ts_trace = accum_overlap.iloc[:,_]
    ts_MoE = 1.96*(
        std_overlap.iloc[:,_] 
        / np.sqrt(
            gdf_overlap['trace_count']
            .astype('float').iloc[_]))

    plt.figure()
    plt.title(core_overlap.columns[_])
    ax = ts_core.plot(
        color='blue', linewidth=2, label='Core')
    ts_trace.plot(
        ax=ax, color='red', linewidth=2, label='PAIPR')
    (ts_trace+ts_MoE).plot(
        ax=ax, color='red', linestyle='--', label='_hidden')
    (ts_trace-ts_MoE).plot(
        ax=ax, color='red', linestyle='--', label='_hidden')
    plt.legend()
    plt.show()

# %%[markdown]
# ### Repeatability of PAIPR results
# 
# OIB has a number of repeat flightlines that allow for testing how reproducible PAIPR results are from different years.
# For these results, we look at two flightlines (11 Nov 2011 and 11 Nov 2016) that have a significant level of overlap to determine if PAIPR produces similar estimates for the same years from both flightlines.
# To ensure adequate spatial coverage, we use the time period 1990-2010 for this investigation.
# In total, we compare time series between the two flights at 1,601 locations.
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

#%%
plt_accum2011 = gv.Points(
    gdf_2011, crs=ANT_proj, vdims=['accum']).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, 
        tools=['hover'], width=600, height=400)
plt_accum2016 = gv.Points(
    gdf_2016, crs=ANT_proj, vdims=['accum']).opts(
        projection=ANT_proj, color='accum', 
        cmap='viridis', colorbar=True, 
        tools=['hover'], width=600, height=400)
plt_accum2011 + plt_accum2016

# %%[markdown]
# The above figure shows the mean accumulation (1990-2010) for the two flightlines (2011 on the left and 2016 on the right). 
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

# %%
plt_trend2011 = gv.Points(
    gdf_2011, crs=ANT_proj, 
    vdims=['trend','t_lb','t_ub']).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm_r', symmetric = True, 
        colorbar=True, 
        tools=['hover'], width=600, height=400)
plt_trend2016 = gv.Points(
    gdf_2016, crs=ANT_proj, 
    vdims=['trend','t_lb','t_ub']).opts(
        projection=ANT_proj, color='trend', 
        cmap='coolwarm_r', colorbar=True, 
        symmetric=True, 
        tools=['hover'], width=600, height=400)
plt_trend2011 + plt_trend2016

# %%[markdown]
# The above figure shows the linear trends 1990-2010 for the two flightlines.
#  
# %%
df_dist = nearest_neighbor(
    gdf_2011, gdf_2016, return_dist=True)
idx_2011 = df_dist['distance'] <= 250
dist_overlap1 = df_dist[idx_2011]

# Create numpy arrays for relevant results
accum_2011 = a2011_ALL.iloc[
    :,dist_overlap1.index].to_numpy()
std_2011 = std2011_ALL.iloc[
    :,dist_overlap1.index].to_numpy()
accum_2016 = a2016_ALL.iloc[
    :,dist_overlap1['trace_ID']].to_numpy()
std_2016 = std2016_ALL.iloc[
    :,dist_overlap1['trace_ID']].to_numpy()

# Create new gdf of subsetted results
gdf_PAIPR = gpd.GeoDataFrame(
    {'ID_2011': dist_overlap1.index, 
    'ID_2016': dist_overlap1['trace_ID'], 
    'accum_2011': accum_2011.mean(axis=0), 
    'accum_2016': accum_2016.mean(axis=0)},
    geometry=dist_overlap1.geometry.values)


# Calculate residuals (as % bias of mean accumulation)
res_PAIPR = (
    (accum_2016 - accum_2011) 
    / np.mean(
        [accum_2011.mean(axis=0), 
        accum_2016.mean(axis=0)], axis=0))

import seaborn as sns
sns.kdeplot(res_PAIPR.reshape(res_PAIPR.size))

print(
    f"The mean bias between 2016 and 2011 "
    f"flights using PAIPR is "
    f"{res_PAIPR.mean()*100:.2f}% "
    f"with a RMSE of {res_PAIPR.std()*100:.2f}%."
)
print(
    f"The mean standard deviations of the annual accumulation "
    f"estimates are "
    f"{(std_2011/accum_2011).mean()*100:.2f}% for " f"2011 and "
    f"{(std_2016/accum_2016).mean()*100:.2f}% for 2016."
)
# %%[markdown]
# The above figure shows the distribution in residuals for 2016 and 2011 PAIPR results, with bias and RMSE statistics.
# This is further compared in the follwing plot, which breaks the comparisons down by year using a 1:1 scatter plot, with 2011 flight results on the x-axis and 2016 results on the y-axis.
# 
#%%

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

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=accum_df, 
    kdims=['accum_2011', 'accum_2016'], 
    vdims=['Year']).groupby('Year')
one_to_one.opts(color='black') * scatt_yr.opts(xlim=(100,750), ylim=(100,750))

# %%[markdown]
# We do a similar comparison as well, but this time for the linear trends.
#
# %%

trend_2011, _, lb_2011, ub_2011 = trend_bs(
    pd.DataFrame(accum_2011, index=a2011_ALL.index), 
    1000, df_err=pd.DataFrame(std_2011, index=std2011_ALL.index))
gdf_PAIPR['trend_2011'] = trend_2011 / gdf_PAIPR['accum_2011']
gdf_PAIPR['lb_2011'] = lb_2011 / gdf_PAIPR['accum_2011']
gdf_PAIPR['ub_2011'] = ub_2011 / gdf_PAIPR['accum_2011']

trend_2016, _, lb_2016, ub_2016 = trend_bs(
    pd.DataFrame(accum_2016, index=a2016_ALL.index), 
    1000, df_err=pd.DataFrame(std_2016, index=std2016_ALL.index))
gdf_PAIPR['trend_2016'] = trend_2016 / gdf_PAIPR['accum_2016']
gdf_PAIPR['lb_2016'] = lb_2016 / gdf_PAIPR['accum_2016']
gdf_PAIPR['ub_2016'] = ub_2016 / gdf_PAIPR['accum_2016']

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[-0.04,0.04], 'y':[-0.04,0.04]}))
scatt_yr = hv.Points(
    data=pd.DataFrame(
        gdf_PAIPR).drop('geometry', axis=1), 
    kdims=['trend_2011', 'trend_2016'])
one_to_one.opts(color='black') * scatt_yr

# %%[markdown]
# We also have a handful of overlapping locations between these two flightlines with SMB time series based on manually-traced layering.
# This permits an investigation of PAIPR repeatability compared to manually-traced layer repeatability in echograms.
# In the case of the manual data, all of the prescribed uncertainty comes from the depth-density model, allowing us to also estimate what fraction of the PAIPR uncertainty comes from PAIPR itself.  
# In these manual data, we have time series at 414 overlapping locations. 
# 
# %% 
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

# Subset manual results to overlapping sections
df_dist = nearest_neighbor(
    gdf_man2016, gdf_man2011, return_dist=True)
idx_2016 = df_dist['distance'] <= 250
dist_overlap = df_dist[idx_2016]

# Create numpy arrays for relevant results
accum_man2011 = (
    man2011_ALL.iloc[:,dist_overlap['trace_ID']].to_numpy())
std_man2011 = (
    manSTD_2011_ALL.iloc[:,dist_overlap['trace_ID']].to_numpy())
accum_man2016 = (
    man2016_ALL.iloc[:,dist_overlap.index].to_numpy())
std_man2016 = (
    manSTD_2011_ALL.iloc[:,dist_overlap.index].to_numpy())

# Create new gdf of subsetted results
gdf_traces = gpd.GeoDataFrame(
    {'ID_2011': dist_overlap['trace_ID'], 
    'ID_2016': dist_overlap.index, 
    'accum_man2011': accum_man2011.mean(axis=0), 
    'accum_man2016': accum_man2016.mean(axis=0)},
    geometry=dist_overlap.geometry.values)

# Assign flight chunk class based on location
chunk_centers = gpd.GeoDataFrame(
    ['mid-accum', 'SEAT2010-4', 'high-accum'],
    geometry=gpd.points_from_xy(
        [-1.177E6, -1.159E6, -1.263E6], 
        [-2.898E5, -4.640E5, -4.868E5]), 
    crs="EPSG:3031")

mid_accum_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(chunk_centers.geometry.x[0], gdf_traces.shape[0]), 
        np.repeat(chunk_centers.geometry.y[0], gdf_traces.shape[0])), 
        crs="EPSG:3031")
    .distance(gdf_traces.reset_index())) <= 30000).values

SEAT4_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(chunk_centers.geometry.x[1], gdf_traces.shape[0]), 
        np.repeat(chunk_centers.geometry.y[1], gdf_traces.shape[0])), 
        crs="EPSG:3031")
    .distance(gdf_traces.reset_index())) <= 30000).values
        
high_accum_idx = ((
    gpd.GeoSeries(gpd.points_from_xy(
        np.repeat(chunk_centers.geometry.x[2], gdf_traces.shape[0]), 
        np.repeat(chunk_centers.geometry.y[2], gdf_traces.shape[0])), 
        crs="EPSG:3031")
    .distance(gdf_traces.reset_index())) <= 30000).values

gdf_traces['Site'] = np.repeat('Null', gdf_traces.shape[0])
gdf_traces['Site'][mid_accum_idx] = 'mid-accum'
gdf_traces['Site'][SEAT4_idx] = 'SEAT2010-4'
gdf_traces['Site'][high_accum_idx] = 'high-accum'

# %%
# Calculate residuals (as % bias of mean accumulation)
man_res = (
    (accum_man2016 - accum_man2011) 
    / np.mean(
        [accum_man2011.mean(axis=0), 
        accum_man2016.mean(axis=0)], axis=0))

sns.kdeplot(man_res.reshape(man_res.size))

print(
    f"The mean bias in manually-derived annual accumulation "
    f"for 2016 vs. 2011 flights is "
    f"{man_res.mean()*100:.2f}% "
    f"with a RMSE of {man_res.std()*100:.2f}% "
)

print(
    f"The mean standard deviations of the annual accumulation "
    f"estimates are "
    f"{(std_man2011/accum_man2011).mean()*100:.2f}% for " f"2011 and "
    f"{(std_man2016/accum_man2016).mean()*100:.2f}% for 2016."
)

# %%[markdown]
# The above plot shows the distribution in residuals for 2011 and 2016 manual results.
# These are quite similar to those for PAIPR, as are the bias and RMSE statistics.
# Furthermore, the mean error associated with the manual results suggests that ~1/2 of the error in PAIPR results from uncertainties in the depth-density estimates. 
# 
# %%
# Create dataframes for scatter plots
accum_man_df = pd.DataFrame(
    {'Trace': np.tile(
        np.arange(0,accum_man2011.shape[1]), 
        accum_man2011.shape[0]), 
    'Site': np.tile(
        gdf_traces['Site'], accum_man2011.shape[0]), 
    'Year': np.reshape(
        np.repeat(man2011_ALL.index, accum_man2011.shape[1]), 
        accum_man2011.size), 
    'accum_2011': 
        np.reshape(accum_man2011, accum_man2011.size), 
    'std_2011': np.reshape(std_man2011, std_man2011.size), 
    'accum_2016': 
        np.reshape(accum_man2016, accum_man2016.size), 
    'std_2016': np.reshape(std_man2016, std_man2016.size)})

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[100,750], 'y':[100,750]}))
scatt_yr = hv.Points(
    data=accum_man_df, 
    kdims=['accum_2011', 'accum_2016'], 
    vdims=['Year']).groupby('Year')
(
    one_to_one.opts(color='black') 
    * scatt_yr.opts(
        xlim=(100,750), ylim=(100,750), 
        xlabel='2011 flight (mm/yr)', 
        ylabel='2016 flight (mm/yr)'))

# %%[markdown]
# The plot above shows a 1:1 plot for annual SMB estimates between 2011 and 2016 results for manually-generated data, again broken out by year.
# The overall variability and bias are similar to PAIPR comparisons, but significant error is introduced from one of the sites (SEAT2010-4).
#  
# %%
T_man2011, intcpt, lb_man2011, ub_man2011 = trend_bs(
    pd.DataFrame(
        accum_man2011, index=man2011_ALL.index), 
    1000, 
    df_err=pd.DataFrame(
        std_man2011, index=man2011_ALL.index))

T_man2016, intcpt, lb_man2016, ub_man2016 = trend_bs(
    pd.DataFrame(
        accum_man2016, index=man2016_ALL.index), 
    1000, 
    df_err=pd.DataFrame(
        std_man2016, index=man2016_ALL.index))

accum_bar = gdf_traces[
    ['accum_man2011', 'accum_man2016']].mean(
        axis=1).values
gdf_traces['trend_2011'] = T_man2011 / accum_bar
gdf_traces['t2011_lb'] = lb_man2011 / accum_bar
gdf_traces['t2011_ub'] = ub_man2011 / accum_bar
gdf_traces['trend_2016'] = T_man2016 / accum_bar
gdf_traces['t2016_lb'] = lb_man2016 / accum_bar
gdf_traces['t2016_ub'] = ub_man2016 / accum_bar
gdf_traces['trend_res'] = (
    T_man2016 - T_man2011) / accum_bar

one_to_one = hv.Curve(
    data=pd.DataFrame(
        {'x':[-0.035,0.01], 'y':[-0.035,0.01]}))

scatt_yr = hv.Points(
    data=pd.DataFrame(gdf_traces), 
    kdims=['trend_2011', 'trend_2016'], 
    vdims=['Site']).groupby('Site')
(
    one_to_one.opts(color='black') 
    * scatt_yr.opts(
        xlim=(-0.035,0.01), ylim=(-0.035,0.01), 
        xlabel='2011 trend (%/yr)', 
        ylabel='2016 flight (%/yr)'))
# %% [markdown]
# A similar 1:1 plot for manual results, but for linear trend estimates, broken out by site.
# Note again that SEAT2010-4 shows significant deviation between the two years (where 2016 estimates much more positive trends than 2011).
# The high accumulation site site also shows more positive results for 2016, but to a lesser degree than SEAT2010-4.
# The mid-accum site shows minimal bias.
# Also note that uncertainties in these trends are high compared to trend magnitudes, with uncertainties of 1-2 percent of the multidecadal mean accumulation. 
# 
# %%
