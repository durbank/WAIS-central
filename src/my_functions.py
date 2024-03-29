# Module script to store project-specific custom functions and variables

# # Import requisite modules
# import numpy as np
# import pandas as pd
# import geopandas as gpd
# from pathlib import Path
# import time
# import shutil
# import os
# import rasterio as rio
# from sklearn.neighbors import BallTree
# from scipy import signal
# import statsmodels.tsa.stattools as tsa
# import richdem as rd

# # Function to import and concatenate PAIPR .csv files
# def import_PAIPR(input_dir):
#     """
#     Function to import PAIPR-derived accumulation data.
#     This concatenates all files within the given directory into a single pandas dataframe.

#     Parameters:
#     input_dir (pathlib.PosixPath): Absolute path of directory containing .csv files of PAIPR results (gamma-distributions fitted to accumulation curves). This directory can contain multiple files, which will be concatenated to a single dataframe.
#     """

#     data = pd.DataFrame()
#     for file in input_dir.glob("*.csv"):
#         data_f = pd.read_csv(file)
#         data = data.append(data_f)
#     data['collect_time'] = pd.to_datetime(
#         data['collect_time'])
#     return data


# # Function to format imported PAIPR data
# def format_PAIPR(data_df, start_yr=None, end_yr=None):
#     """

#     """
#     # Create groups based on trace locations
#     traces = data_df.groupby(
#         ['collect_time', 'Lat', 'Lon'])
#     data_df = data_df.assign(trace_ID = traces.ngroup())
#     traces = data_df.groupby('trace_ID')

#     if start_yr != end_yr:
#         # Remove time series with data missing from period
#         # of interest (and clip to period of interest)
#         data_df = traces.filter(
#             lambda x: min(x['Year']) <= start_yr 
#             and max(x['Year']) >= end_yr)
#         data_df = data_df.query(
#             f"Year >= {start_yr} & Year <= {end_yr}")

#     # # Ensure each trace has only one time series 
#     # # (if not, take the mean of all time series)
#     # data = data.groupby(['trace_ID', 'Year']).mean()

#     if 'gamma_shape' in data_df.columns:
#         # Generate descriptive statistics based on 
#         # imported gamma-fitted parameters
#         alpha = data_df['gamma_shape']
#         alpha.loc[alpha<1] = 1
#         beta = 1/data_df['gamma_scale']
#         mode_accum = (alpha-1)/beta
#         var_accum = alpha/beta**2
        
#         # New df (in long format) with accum data assigned
#         data_long = (
#             data_df.filter(['trace_ID', 'collect_time', 
#             'QC_flag', 'Lat', 'Lon', 
#             'elev', 'Year']).assign(
#                 accum=mode_accum, 
#                 std=np.sqrt(var_accum))
#             .reset_index(drop=False))
#     else:
#         # New df (in long format) with accum data assigned
#         data_long = (
#             data_df.filter(['trace_ID', 'collect_time', 
#             'QC_flag', 'QC_med', 'Lat', 'Lon', 
#             'elev', 'Year']).assign(
#                 accum=data_df['accum_mu'], 
#                 std=data_df['accum_std']).reset_index(drop=True))

#     # Additional subroutine to remove time series where the deepest 
#     # 3 years have overly large uncertainties (2*std > expected value)
#     # data_tmp = data_long[data_long['Year'] <= (start_yr+2)]
#     # data_log = pd.DataFrame(
#     #     {'trace_ID': data_tmp['trace_ID'], 
#     #     'ERR_log': 2*data_tmp['std'] > data_tmp['accum']})
#     # trace_tmp = data_log.groupby('trace_ID')
#     # IDs_keep = trace_tmp.filter(
#     #     lambda x: not all(x['ERR_log']))['trace_ID'].unique()
#     data_tmp = data_long.groupby(
#         'trace_ID').mean().query('2*std < accum').reset_index()
#     IDs_keep = data_tmp['trace_ID']
#     data_long = data_long[
#         data_long['trace_ID'].isin(IDs_keep)]

#     # Remove time series with fewer than 5 years
#     data_tmp = data_long.join(
#         data_long.groupby('trace_ID')['Year'].count(), 
#         on='trace_ID', rsuffix='_count')
#     data_long = data_tmp.query('Year_count >= 5').drop(
#         'Year_count', axis=1)

#     # Reset trace IDs to match total number of traces
#     tmp_group = data_long.groupby('trace_ID')
#     data_long['trace_ID'] = tmp_group.ngroup()
#     data_final = data_long.sort_values(
#         ['trace_ID', 'Year']).reset_index(drop=True)

#     return data_final

# def long2gdf(accum_long):
#     """
#     Function to convert data in long format to geodataframe aggregated
#     by trace location and sorted by collection time
#     """

#     accum_long['collect_time'] = (
#         accum_long.collect_time.values.astype(np.int64))
#     traces = accum_long.groupby('trace_ID').mean().drop(
#         'Year', axis=1)
#     traces['collect_time'] = (
#         pd.to_datetime(traces.collect_time)
#         .dt.round('1ms'))

#     # Sort by collect_time and reset trace_ID index
#     # traces = (traces.sort_values('collect_time')
#     #     .reset_index(drop=True))
#     # traces.index.name = 'trace_ID'
#     traces = traces.reset_index()

#     gdf_traces = gpd.GeoDataFrame(
#         traces.drop(['Lat', 'Lon'], axis=1), 
#         geometry=gpd.points_from_xy(
#             traces.Lon, traces.Lat), 
#         crs="EPSG:4326")

#     return gdf_traces

# def pts2grid(geo_df, resolution=5000):
#     import shapely
#     """
#     Description.
#     """
#     xmin, ymin, xmax, ymax = geo_df.total_bounds

#     x_pts = np.arange(
#         np.floor(xmin), np.ceil(xmax)+resolution, 
#         step=resolution)
#     y_pts = np.arange(
#         np.floor(ymin), np.ceil(ymax)+resolution, 
#         step=resolution)

#     grid_cells = []
#     for x in x_pts:
#         for y in y_pts:
#             x0 = x - resolution
#             y0 = y - resolution
#             grid_cells.append(shapely.geometry.box(x0, y0, x, y))
    
#     geo_cells = gpd.GeoDataFrame(
#         data=grid_cells, columns=['geometry'],
#         crs=geo_df.crs)
#     # # Diagnostic plot
#     # ax = geo_cells.plot(facecolor='none', edgecolor='grey')
#     # geo_df.plot(ax=ax, column='accum', cmap='viridis', vmax=600)

#     geo_sjoin = gpd.sjoin(
#         geo_df, geo_cells, how='inner', op='within')
#     gdf_grid = gpd.GeoDataFrame(
#         data=geo_sjoin[['trace_ID', 'elev', 'accum', 'std', 'index_right']], 
#         geometry=geo_cells.geometry[geo_sjoin['index_right']].values, 
#         crs=geo_df.crs)
    
#     return gdf_grid


# def trace_combine(gdf_grid, accum_ALL, std_ALL):
#     """
#     Description.
#     """
#     # Get indices of grids that are populated
#     grid_pop = np.unique(gdf_grid['index_right'])

#     # Preallocate return arrays
#     accum_df = pd.DataFrame(
#         columns=np.arange(len(grid_pop)), 
#         index=accum_ALL.index)
#     MoE_df = pd.DataFrame(
#         columns=np.arange(len(grid_pop)), 
#         index=accum_ALL.index)
#     yr_count = pd.DataFrame(
#         columns=np.arange(len(grid_pop)), 
#         index=accum_ALL.index)
#     grid_final = gpd.GeoDataFrame(
#         columns=[
#             'grid_ID', 'elev', 'trace_count', 
#             'accum', 'MoE', 'geometry'], 
#         index=np.arange(len(grid_pop)), crs=gdf_grid.crs)
    

#     for i, grid_idx in enumerate(grid_pop):

#         # Subset geodf to traces in current grid and get trace_IDs
#         gdf_tmp = gdf_grid[gdf_grid['index_right']==grid_idx]
#         t_IDs = gdf_tmp['trace_ID'].values

#         # Get subsetted accum and std arrays
#         accum_arr = accum_ALL.iloc[:,t_IDs]
#         std_arr = std_ALL.iloc[:,t_IDs]

#         nsim = 10
#         accum_sim = pd.DataFrame(np.zeros(
#             (accum_arr.shape[0], nsim*accum_arr.shape[1])), 
#             index=accum_arr.index)
#         for j in range(accum_arr.shape[1]):

#             col_mu = accum_arr.iloc[:,j]
#             col_std = std_arr.iloc[:,j]

#             for k in range(nsim):
#                 accum_sim.iloc[:,(j*nsim+k)] = np.random.normal(
#                     loc=col_mu, scale=col_std)

#         accum_mu = accum_sim.mean(axis=1)
#         n_count = accum_sim.count(axis=1)
#         accum_moe = (
#             1.96*accum_sim.std(axis=1) 
#             / np.sqrt(n_count))

#         # # Calculate weighted mean accum and std
#         # weights = (1/std_arr) / np.tile(
#         #     (1/std_arr).sum(axis=1).to_numpy(), 
#         #     (std_arr.shape[1],1)). transpose()
#         # accum_w = (accum_arr*weights).sum(axis=1)
#         # std_w = np.sqrt(((std_arr**2)*weights).sum(axis=1))
#         # accum_mu = accum_w
#         # accum_moe = (
#         #     1.96*std_w
#         #     / np.sqrt(accum_arr.count(axis=1)))

#         # Set missing data to NaN
#         nan_idx = np.invert(
#             accum_arr.sum(axis=1).astype('bool'))
#         accum_mu[nan_idx] = np.nan
#         accum_moe[nan_idx] = np.nan

#         # Add weighted mean and std to return arrays
#         accum_df.iloc[:,i] = accum_mu
#         MoE_df.iloc[:,i] = accum_moe

#         # Get counts for number of records in annual 
#         # estimates and add to return array
#         tmp_yr = accum_arr.count(axis=1)
#         tmp_yr[nan_idx] = np.nan
#         yr_count.iloc[:,i] = tmp_yr

#         # Populate return geodf
#         grid_final.loc[i,'grid_ID'] = grid_idx
#         grid_final.loc[i,'elev'] = gdf_tmp['elev'].mean()
#         grid_final.loc[i, 'trace_count'] = len(t_IDs)
#         grid_final.loc[i,'accum'] = accum_mu.mean()
#         grid_final.loc[i,'MoE'] = (
#             1.96*accum_mu.std()/np.sqrt(accum_mu.count()))
#         grid_final.loc[i,'geometry'] = gdf_tmp.iloc[0].geometry

#     return grid_final, accum_df, MoE_df, yr_count

# def get_nearest(
#     src_points, candidates, k_neighbors=1):
#     """Find nearest neighbors for all source points from a set of candidate points"""

#     # Create tree from the candidate points
#     tree = BallTree(candidates, leaf_size=15, metric='haversine')

#     # Find closest points and distances
#     distances, indices = tree.query(src_points, k=k_neighbors)

#     # Transpose to get distances and indices into arrays
#     distances = distances.transpose()
#     indices = indices.transpose()

#     if k_neighbors==2:
#         # Select 2nd closest (as first closest will be the same point)
#         closest = indices[1]
#         closest_dist = distances[1]
#     else:
#         # Get closest indices and distances (i.e. array at index 0)
#         # note: for the second closest points, you would take index 1, etc.
#         closest = indices[0]
#         closest_dist = distances[0]

#     # Return indices and distances
#     return (closest, closest_dist)


# def nearest_neighbor(left_gdf, right_gdf, return_dist=False, planet_radius=6371000):
#     """
#     For each point in left_gdf, find closest point in right GeoDataFrame and return them.
#     When return_dist=True, also returns the distance between the nearest points for each entry.
#     The planet radius (in meters) defaults to Earth's radius. 

#     NOTICE: Assumes that the input Points are in WGS84 projection (lat/lon).
#     """

#     end_crs = left_gdf.crs

#     # Ensures data are in WGS84 projection
#     left_gdf = left_gdf.to_crs('EPSG:4326')
#     right_gdf = right_gdf.to_crs('EPSG:4326')

#     left_geom_col = left_gdf.geometry.name
#     right_geom_col = right_gdf.geometry.name

#     # Ensure that index in right gdf is formed of sequential numbers
#     right = right_gdf.copy().reset_index(drop=True)

#     # Parse coordinates from points and insert them into a numpy array as RADIANS
#     left_radians = np.array(
#         left_gdf[left_geom_col].apply(
#         lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
#     right_radians = np.array(
#         right[right_geom_col].apply(
#         lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())

#     # Find the nearest points
#     # -----------------------
#     # closest ==> index in right_gdf that corresponds to the closest point
#     # dist ==> distance between the nearest neighbors (in meters)

#     closest, dist = get_nearest(
#         src_points=left_radians, 
#         candidates=right_radians)

#     # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
#     closest_points = right.loc[closest]

#     # Ensure that the index corresponds the one in left_gdf
#     closest_points = closest_points.reset_index(drop=True)

#     # Add distance if requested
#     if return_dist:
#         # Convert to meters from radians
#         closest_points['distance'] = dist * planet_radius

#     # Reproject results back to original projection
#     closest_points = closest_points.to_crs(end_crs)

#     return closest_points


# def get_Xovers(
#     gdf_Xover, gdf_candidates, cutoff_dist):
#     """
#     Function to extract crossover locations (within a cutoff distance) and return traceID values at those locations. 

#     NOTE: Assumes that input gdf is for Antarctica.
#     """

#     gdf_Xover = gdf_Xover.to_crs("EPSG:3031")
#     gdf_candidates = gdf_candidates.to_crs(
#         "EPSG:3031")
    
#     gdf_Xover['geometry'] = gdf_Xover.buffer(
#         cutoff_dist)

#     candidate_near = gpd.sjoin(
#         gdf_candidates, gdf_Xover).drop(
#             columns=['index_right'])
    
#     Xover_idx1 = candidate_near.collect_time.argmax()
#     Xover_idx2 = candidate_near.collect_time.argmin()
#     trace_ID1 = candidate_near['trace_ID'].iloc[
#         Xover_idx1]
#     trace_ID2 = candidate_near['trace_ID'].iloc[
#         Xover_idx2]

#     t_delta = (
#         candidate_near.collect_time.max() 
#         - candidate_near.collect_time.min()
#     )
#     print(f"Time difference between traces is {t_delta}")

#     return trace_ID1, trace_ID2


# import holoviews as hv
# hv.extension('bokeh')
# def plot_Xover(
#     accum_data, std_data, ts_trace1, ts_trace2):

#     """
#     Function to generate plot objects comparing the time series of two cross-over locations of the same flightline.
#     """

#     ts1 = pd.DataFrame(
#         {'accum': accum_data[ts_trace1], 
#         'upper': accum_data[ts_trace1] 
#             + std_data[ts_trace1], 
#         'lower': accum_data[ts_trace1] 
#             - std_data[ts_trace1]})
#     ts2 = pd.DataFrame(
#         {'accum': accum_data[ts_trace2], 
#         'upper': accum_data[ts_trace2] 
#             + std_data[ts_trace2], 
#         'lower': accum_data[ts_trace2] 
#             - std_data[ts_trace2]})

#     plt_ts1 = (
#         hv.Curve(data=ts1, 
#             kdims=['Year', 'accum']).opts(
#             color='blue', line_width=2) 
#         * hv.Curve(data=ts1, 
#         kdims=['Year','upper']).opts(
#             color='blue', line_dash='dotted') 
#         * hv.Curve(data=ts1, 
#             kdims=['Year','lower']).opts(
#             color='blue', line_dash='dotted'))
#     plt_ts2 = (
#         hv.Curve(data=ts2, 
#             kdims=['Year', 'accum']).opts(
#             color='red', line_width=2) 
#         * hv.Curve(data=ts2, 
#         kdims=['Year','upper']).opts(
#             color='red', line_dash='dotted') 
#         * hv.Curve(data=ts2, 
#             kdims=['Year','lower']).opts(
#             color='red', line_dash='dotted'))
    
#     return plt_ts1, plt_ts2

# def trend_bs(df, nsim, df_err=pd.DataFrame()):
#     """
#     Dpc string goes here.
#     """
#     tic = time.perf_counter()
    
#     # Preallocate results arrays
#     trends_bs = pd.DataFrame(columns=df.columns)
#     intercepts = pd.DataFrame(columns=df.columns)

#     # In no errors exist, create neutral weights
#     if df_err.empty:
#         df_err = pd.DataFrame(
#             np.ones(df.shape), index=df.index, 
#             columns=df.columns)

#     # Peform bootstrapping
#     for _ in range(nsim):
#         # Randomly resample data
#         data_bs = df.sample(
#             len(df), replace=True).sort_index()
        
#         # Generate mean weights
#         weights_bs = (
#                 1/df_err.loc[data_bs.index]).mean(axis=1)

#         # Check for nans (requires for loop w/ 2D array)
#         if np.any(np.isnan(data_bs)):
            
#             data_ma = np.ma.array(data_bs, mask=np.isnan(data_bs))
#             coeffs = np.zeros((2,data_ma.shape[1]))
#             for idx in range(coeffs.shape[1]):
#                 data_i = data_ma[:,idx]
#                 if (np.invert(data_i.mask)).sum() < 5:
#                     coeffs[:,idx] = np.nan
#                 else:
#                     coeffs[:,idx] = np.ma.polyfit(
#                         data_bs.index, data_i, 1, w=weights_bs)
                
        
#         else:
#             # If no nan's present, perform vectorized fitting
#             coeffs = np.polyfit(
#                 data_bs.index, data_bs, 1, w=weights_bs)

#         trends_bs = trends_bs.append(
#             pd.Series(coeffs[0], index=df.columns), 
#             ignore_index=True)
#         intercepts = intercepts.append(
#             pd.Series(coeffs[1], index=df.columns), 
#             ignore_index=True)

#     toc = time.perf_counter()
#     print(f"Execution time of bootstrapping: {toc-tic} s")

#     trend_mu = np.nanmean(trends_bs, axis=0)
#     intercept_mu = np.nanmean(intercepts, axis=0)
#     trendCI_lb = np.nanpercentile(trends_bs, 2.5, axis=0)
#     trendCI_ub = np.nanpercentile(trends_bs, 97.5, axis=0)

#     return trend_mu, intercept_mu, trendCI_lb, trendCI_ub

# # Function to perform autocorrelation
# def acf(df):
#     """
#     Doc string goes here.
#     """
#     # Detrend time series data
#     arr_ts = signal.detrend(df, axis=0)

#     lags = int(np.round(arr_ts.shape[0]/2))
#     arr_acf = np.zeros((lags, arr_ts.shape[1]))
#     for idx, col in enumerate(arr_ts.T):
#         arr_acf[:,idx] = tsa.acf(col, nlags=lags-1)
#     acf_df = pd.DataFrame(
#         arr_acf, columns=df.columns, 
#         index=np.arange(lags))
#     acf_df.index.name = 'Lag'

#     return acf_df

# import requests
# def get_REMA(tile_idx, output_dir):
#     """
#     Downloads, unzips, and saves DEM tiles from the Reference Elevation Model of Antarctica (REMA) dataset.
#     Required inputs:
#     tile_idx {pandas.core.frame.DataFrame}: Dataframe with the names (name), tile IDs (tile), and file urls (fileurl) of the requested data for download, taken from the [REMA tile shapefile](http://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/).
#     output_dir {pathlib.PosixPath}: Path of directory at which to download the requested file.
#     """

#     for idx, row in tile_idx.iterrows():
#         f_dir = output_dir.joinpath(row.tile)

#         if not f_dir.exists():
#             f_dir.mkdir(parents=True)
#             zip_path = f_dir.joinpath('tmp.tar.gz')
#             r = requests.get(row.fileurl, stream=True)
#             print(f"Downloading tile {f_dir.name}")
#             with open(zip_path, 'wb') as zFile:
#                 for chunk in r.iter_content(
#                         chunk_size=1024*1024):
#                     if chunk:
#                         zFile.write(chunk)
#             print(f"Unzipping tile {f_dir.name}")
#             shutil.unpack_archive(zip_path, f_dir)
#             os.remove(zip_path)
#         else:
#             print(f"REMA tile {f_dir.name} already exists locally, moving to next download")
#     print("All requested files downloaded")


# def calc_topo(dem_path):
#     """
#     Calculates slope and aspect from given DEM and saves output.
#     The function checks to see whether a slope/aspect file has already been created so as to avoid needless processing.
    
#     Parameters:
#     dem_path (pathlib.PosixPath): The relative or absolute path to an input DEM file.

#     Dependencies: 
#     richdem module
#     GDAL binaries
#     pathlib module
#     """
#     slope_path = Path(
#         str(dem_path).replace("dem", "slope"))
#     aspect_path = Path(
#         str(dem_path).replace("dem", "aspect"))

#     if ((not slope_path.is_file()) or 
#             (not aspect_path.is_file())):
        
#         # Load DEM
#         dem = rd.LoadGDAL(str(dem_path))
        
#         # Calculate slope values from DEM
#         if not slope_path.is_file():
#             print(f"Calculating slope values for REMA tile {dem_path.parent.name}...")
#             slope = rd.TerrainAttribute(
#                 dem, attrib='slope_riserun')
#             rd.SaveGDAL(str(slope_path), slope)
#         else:
#                 print(f"Slope data already exist locally for REMA tile {dem_path.parent.name}. Checking for aspect data...")
        
#         # Calculate aspect values from DEM
#         if not aspect_path.is_file():
#             print(f"Calculating aspect values for REMA tile {dem_path.parent.name}...")
#             aspect = rd.TerrainAttribute(dem, attrib='aspect')
#             rd.SaveGDAL(str(aspect_path), aspect)
#         else:
#             print(f"Aspect data already exist locally for REMA tile {dem_path.parent.name}. Moving to next tile...")

#     else:
#         print(f"Slope/aspect geotifs already exist locally for REMA tile {dem_path.parent.name}. Moving to next tile...")


# def topo_vals(tile_dir, locations, slope=False, aspect=False):
#     """Extracts elevation, slope, and aspect values at given locations.
#     Dependencies: Requires the rasterio (as rio) module and, by extension, GDAL binaries.
#     Requires the geopandas module.

#     Args:
#         tile_dir (pathlib.PosixPath): The relative or absolute path to a directory containing REMA tile DSM, slope and aspect geotiffs.
#         locations (geopandas.geodataframe.GeoDataFrame): A geodataframe containing the locations at which to extract raster data. These data should have the geometries stored in a column named "geometry" (the default for geopandas).
#         slope (bool, optional): Whether to also extract slope values at points of interest. Defaults to False.
#         aspect (bool, optional): Whether to also extract aspect values at points of interest. Defaults to False.

#     Returns:
#         geopandas.geodataframe.GeoDataFrame: The same input geodataframe with topographic values appended.
#     """

#     # Create empty Elev column if missing in gdf
#     if 'elev' not in locations.columns:
#         elev = np.empty(locations.shape[0])
#         elev[:] = np.NaN
#         locations['elev'] = elev

#     # Get initial gdf crs
#     crs_init = locations.crs

#     # Ensure locations are in same crs as REMA (EPSG:3031)
#     locations.to_crs(epsg=3031, inplace=True)

#     # Extract coordinates of sample points
#     coords = (
#         [(x,y) for x, y in zip(
#             locations.geometry.x, locations.geometry.y)]
#     )

#     # Extract elevation values for all points within tile
#     tile_path = [
#         file for file in tile_dir.glob("*dem.tif")][0]
#     src = rio.open(tile_path)
#     tile_vals = np.asarray(
#         [x[0] for x in src.sample(coords, masked=True)])
#     tile_mask = ~np.isnan(tile_vals)
#     locations.loc[tile_mask,'elev'] = tile_vals[tile_mask]
#     src.close()

#     # Force elevation data to numeric
#     #

#     if slope:
#         # Create empty slope column if missing in gdf
#         if 'slope' not in locations.columns:
#             slope = np.empty(locations.shape[0])
#             slope[:] = np.NaN
#             locations['slope'] = slope

#         # Extract slope values for all points within tile
#         tile_path = [
#             file for file in tile_dir.glob("*slope.tif")][0]
#         src = rio.open(tile_path)
#         tile_vals = np.asarray(
#             [x[0] for x in src.sample(coords, masked=True)])
#         tile_mask = ~np.isnan(tile_vals)
#         locations.loc[tile_mask,'slope'] = tile_vals[tile_mask]
#         src.close()

#     if aspect:
#         # Create empty aspect column if missing in gdf
#         if 'aspect' not in locations.columns:
#             aspect = np.empty(locations.shape[0])
#             aspect[:] = np.NaN
#             locations['aspect'] = aspect

#         # Extract aspect values for all points within tile
#         tile_path = [
#             file for file in tile_dir.glob("*aspect.tif")][0]
#         src = rio.open(tile_path)
#         tile_vals = np.asarray(
#             [x[0] for x in src.sample(coords, masked=True)])
#         tile_mask = ~np.isnan(tile_vals)
#         locations.loc[tile_mask,'aspect'] = tile_vals[tile_mask]
#         src.close()

#     # Convert gdf crs back to original
#     locations.to_crs(crs_init, inplace=True)

#     return locations



# def extract_at_pts(
#     xr_ds, gdf_pts, coord_names=['lon','lat'], 
#     return_dist=False, planet_radius=6371000):
#     """
#     Function where, given an xr-dataset and a Point-based geodataframe, extract all values of variables in xr-dataset at pixels nearest the given points in the geodataframe.
#     xr_ds {xarray.core.dataset.Dataset}: Xarray dataset containing variables to extract.
#     gdf_pts {geopandas.geodataframe.GeoDataFrame} : A Points-based geodataframe containing the locations at which to extract xrarray variables.
#     coord_names {list}: The names of the longitude and latitude coordinates within xr_ds.
#     return_dist {bool}: Whether function to append the distance (in meters) between the given queried points and the nearest raster pixel centroids. 
#     NOTE: This assumes the xr-dataset includes lon/lat in the coordinates 
#     (although they can be named anything, as this can be prescribed in the `coord_names` variable).
#     """

#     # Convert xr dataset to df and extract coordinates
#     xr_df = xr_ds.to_dataframe().reset_index()
#     xr_coord = xr_df[coord_names]

#     # Ensure gdf_pts is in lon/lat and extract coordinates
#     crs_end = gdf_pts.crs 
#     gdf_pts.to_crs(epsg=4326, inplace=True)
#     pt_coord = pd.DataFrame(
#         {'Lon': gdf_pts.geometry.x, 
#         'Lat': gdf_pts.geometry.y}).reset_index(drop=True)

#     # Convert lon/lat points to RADIANS for both datasets
#     xr_coord = xr_coord*np.pi/180
#     pt_coord = pt_coord*np.pi/180

#     # Find xr data nearest given points
#     xr_idx, xr_dist = get_nearest(pt_coord, xr_coord)

#     # Drop coordinate data from xr (leaves raster values)
#     cols_drop = list(dict(xr_ds.coords).keys())
#     xr_df_filt = xr_df.iloc[xr_idx].drop(
#         cols_drop, axis=1).reset_index(drop=True)
    
#     # Add raster values to geodf
#     gdf_return = gdf_pts.reset_index(
#         drop=True).join(xr_df_filt)
    
#     # Add distance between raster center and points to gdf
#     if return_dist:
#         gdf_return['dist_m'] = xr_dist * planet_radius
    
#     # Reproject results back to original projection
#     gdf_return.to_crs(crs_end, inplace=True)

#     return gdf_return

    