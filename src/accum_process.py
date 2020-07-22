# %%[markdown]
# # Script to process PAIPR-generated results in Python

# %%

# Import requisite modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import time
import statsmodels.api as SM
import statsmodels.formula.api as sm

# Set project root directory
ROOT_DIR = Path(__file__).parent.parent

# Set project data directory
DATA_DIR = ROOT_DIR.joinpath('data')

# %%
##########
## Custom function definitions

# Function to import and concatenate PAIPR .csv files
def import_PAIPR(input_dir):
    """
    Function to import PAIPR-derived accumulation data.
    This concatenates all files within the given directory into a single pandas dataframe.

    Parameters:
    input_dir (pathlib.PosixPath): Absolute path of directory containing .csv files of PAIPR results (gamma-distributions fitted to accumulation curves). This directory can contain multiple files, which will be concatenated to a single dataframe.
    """

    data = pd.DataFrame()
    for file in input_dir.glob("*.csv"):
        data_f = pd.read_csv(file)
        data = data.append(data_f)
    return data


# Function to format imported PAIPR data
def format_PAIPR(data_raw, start_yr, end_yr):
    """

    """
    # Remove time series with data missing from period
    # of interest (and clip to period of interest)
    traces = data_raw.groupby(['Lat', 'Lon', 'elev'])
    data = data_raw.assign(trace_ID = traces.ngroup())
    traces = data.groupby('trace_ID')
    data = traces.filter(
        lambda x: min(x['Year']) <= start_yr 
        and max(x['Year']) >= end_yr)
    data = data.query(
        f"Year >= {start_yr} & Year <= {end_yr}")

    # Ensure each trace has only one time series 
    # (if not, take the mean of all time series)
    data = data.groupby(['trace_ID', 'Year']).mean()

    # Generate descriptive statistics based on imported 
    # gamma-fitted parameters
    alpha = data['gamma_shape']
    alpha.loc[alpha<1] = 1
    beta = 1/data['gamma_scale']
    mode_accum = (alpha-1)/beta
    var_accum = alpha/beta**2

    # New df (in long format) with accum data assigned
    data_long = (
        data.filter(['trace_ID', 'QC_flag', 'Lat', 
        'Lon', 'elev', 'Year']).assign(
            accum=mode_accum, std=np.sqrt(var_accum))
        .reset_index()
    )
    return data_long

##########

# %%
# # Import PAIPR-generated data
# PAIPR_dir = DATA_DIR.joinpath('gamma_20111109')
# data_0 = import_PAIPR(PAIPR_dir)
# PAIPR_dir = DATA_DIR.joinpath('gamma_20101119')
# data_0 = data_0.append(import_PAIPR(PAIPR_dir))

# Import PAIPR-generated data
data_list = [dir for dir in DATA_DIR.glob('gamma/*/')]
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

# Import Antarctic outline shapefile
ant_path = ROOT_DIR.joinpath(
    'data/Ant_basemap/Coastline_medium_res_polygon.shp')
ant_outline = gpd.read_file(ant_path)

# Convert accum crs to same as Antarctic outline
accum_trace = accum_trace.to_crs(ant_outline.crs)

# %%

# ## Estimate time series regressions

# # Preallocate arrays for linear regression
# lm_data = accum.transpose()
# std_data = accum_std.transpose()
# coeff = np.zeros(lm_data.shape[0])
# std_err = np.zeros(lm_data.shape[0])
# p_val = np.zeros(lm_data.shape[0])
# R2 = np.zeros(lm_data.shape[0])


# # Full stats (with diagnostics) OLS model
# tic = time.perf_counter()

# for i in range(lm_data.shape[0]):
#     X = SM.add_constant(lm_data.columns)
#     y = lm_data.iloc[i]
#     model = SM.OLS(y, X)
#     results = model.fit()
#     coeff[i] = results.params[1]
#     std_err[i] = results.bse[1]
#     p_val[i] = results.pvalues[1]
#     R2[i] = results.rsquared
# toc = time.perf_counter()
# print(f"Execution time of OLS: {toc-tic}s")



# # Full stats (with diagnostics) WLS model
# tic = time.perf_counter()
# for i in range(lm_data.shape[0]):
#     X = SM.add_constant(lm_data.columns)
#     y = lm_data.iloc[i]
#     w = 1/(std_data.iloc[i] ** 2)
#     model = SM.WLS(y, X, weights=w)
#     results = model.fit()
#     coeff[i] = results.params[1]
#     std_err[i] = results.bse[1]
#     p_val[i] = results.pvalues[1]
#     R2[i] = results.rsquared
# toc = time.perf_counter()
# print(f"Execution time of WLS: {toc-tic}s")



# # Full stats (with diagnostics) Robust OLS model
# tic = time.perf_counter()
# for i in range(lm_data.shape[0]):
#     X = SM.add_constant(lm_data.columns)
#     y = lm_data.iloc[i]
#     model = SM.RLM(y, X, M=SM.robust.norms.HuberT())
#     results = model.fit()
#     coeff[i] = results.params[1]
#     std_err[i] = results.bse[1]
#     p_val[i] = results.pvalues[1]
#     # R2_r[i] = results.rsquared
# toc = time.perf_counter()
# print(f"Execution time of RLS: {toc-tic} s")

# # Add regression results to gdf
# accum_trace['trnd'] = coeff
# accum_trace['p_val'] = p_val
# accum_trace['trnd_perc'] = accum_trace.trnd/accum_trace.accum
# accum_trace['std_err'] = std_err / accum_trace.accum

# ## Large-scale spatial multi-linear regression

# # Create normal df of results
# accum_df = pd.DataFrame(accum_trace.drop(columns='geometry'))
# accum_df['East'] = accum_trace.geometry.x
# accum_df['North'] = accum_trace.geometry.y



# %%

# ## Sensitivity tests
# # This section determines how sensitive the calculated linear 
# # regression is to the duration of time selected

# years = accum.index
# trends_span = pd.DataFrame(
#     columns=accum.columns)

# for i in range(len(years)-15):
#     coeffs = np.polyfit(years[i:], accum[i:], 1)
#     trends_span = trends_span.append(
#         pd.Series(coeffs[0], index=accum.columns), 
#         ignore_index=True)


# trends_span.index = np.arange(
#     years[0], (years[0]+len(years)-15))
# trends_span.index.name = 'Year_start'

# trends_span.apply(np.std).plot.kde()

# trends_span.apply(np.mean).plot.kde()


# %%

# Use polyfit and bootstrapping to get mean trend estimate 
# with uncertainty bounds (plus robust to outliers)

tic = time.perf_counter()

trends_bs = pd.DataFrame(columns=accum.columns)
weights = accum_std.median(axis=1)
weights.name = 'weights'

for _ in range(500):
    accum_bs = accum.sample(
        len(accum), replace=True).sort_index()
    # weights_bs = weights[accum_bs.index]
    # coeffs = np.polyfit(
    #     accum_bs.index, accum_bs, 1, w=1/weights_bs)
    coeffs = np.polyfit(accum_bs.index, accum_bs, 1)
    trends_bs = trends_bs.append(
        pd.Series(coeffs[0], index=accum.columns), 
        ignore_index=True)

toc = time.perf_counter()

print(f"Execution time of bootstrapping: {toc-tic} s")

trend_mu = np.mean(trends_bs)
trend_lb = np.percentile(trends_bs, 2.5, axis=0)
trend_ub = np.percentile(trends_bs, 97.5, axis=0)

accum_trace['trend'] = trend_mu
accum_trace['sig'] = ~np.logical_and(
    (trend_lb <= 0), (trend_ub >= 0))
accum_trace['trnd_perc'] = accum_trace.trend/accum_trace.accum
accum_trace['trnd_lb'] = trend_lb/accum_trace.accum
accum_trace['trnd_ub'] = trend_ub/accum_trace.accum






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

# Define data plotting subset (to aid rendering times)
xmin, ymin, xmax, ymax = accum_trace.total_bounds
accum_subset = (accum_trace.cx[-1.42E6:xmax, ymin:ymax]
    .sample(5000)).sort_index()

# %%
## Plot data inset map
Ant_bnds = gv.Shape.from_shapefile(shp, crs=ANT_proj).opts(
    projection=ANT_proj, width=500, height=500)
trace_plt = gv.Points(accum_subset, crs=ANT_proj).opts(
    projection=ANT_proj, color='red')
Ant_bnds * trace_plt

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


# %%
