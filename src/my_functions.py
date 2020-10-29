# Module script to store project-specific custom functions and variables

# Import requisite modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import time
from sklearn.neighbors import BallTree
from scipy import signal
import statsmodels.tsa.stattools as tsa
import holoviews as hv
hv.extension('bokeh')

# # Set project root directory
# ROOT_DIR = Path(__file__).parent.parent

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
    data['collect_time'] = pd.to_datetime(
        data['collect_time'])
    return data


# Function to format imported PAIPR data
def format_PAIPR(data_df, start_yr=None, end_yr=None):
    """

    """
    # Create groups based on trace locations
    traces = data_df.groupby(
        ['collect_time', 'Lat', 'Lon'])
    data_df = data_df.assign(trace_ID = traces.ngroup())
    traces = data_df.groupby('trace_ID')

    if start_yr != end_yr:
        # Remove time series with data missing from period
        # of interest (and clip to period of interest)
        data_df = traces.filter(
            lambda x: min(x['Year']) <= start_yr 
            and max(x['Year']) >= end_yr)
        data_df = data_df.query(
            f"Year >= {start_yr} & Year <= {end_yr}")

    # # Ensure each trace has only one time series 
    # # (if not, take the mean of all time series)
    # data = data.groupby(['trace_ID', 'Year']).mean()

    if 'gamma_shape' in data_df.columns:
        # Generate descriptive statistics based on 
        # imported gamma-fitted parameters
        alpha = data_df['gamma_shape']
        alpha.loc[alpha<1] = 1
        beta = 1/data['gamma_scale']
        mode_accum = (alpha-1)/beta
        var_accum = alpha/beta**2
        
        # New df (in long format) with accum data assigned
        data_long = (
            data_df.filter(['trace_ID', 'collect_time', 
            'QC_flag', 'Lat', 'Lon', 
            'elev', 'Year']).assign(
                accum=mode_accum, 
                std=np.sqrt(var_accum))
            .reset_index(drop=False))
    else:
        # New df (in long format) with accum data assigned
        data_long = (
            data_df.filter(['trace_ID', 'collect_time', 
            'QC_flag', 'Lat', 'Lon', 
            'elev', 'Year']).assign(
                accum=data_df['accum_mu'], 
                std=data_df['accum_std']).reset_index(drop=True))

    # Additional subroutine to remove time series where the deepest 
    # 3 years have overly large uncertainties (2*std > expected value)
    # data_tmp = data_long[data_long['Year'] <= (start_yr+2)]
    # data_log = pd.DataFrame(
    #     {'trace_ID': data_tmp['trace_ID'], 
    #     'ERR_log': 2*data_tmp['std'] > data_tmp['accum']})
    # trace_tmp = data_log.groupby('trace_ID')
    # IDs_keep = trace_tmp.filter(
    #     lambda x: not all(x['ERR_log']))['trace_ID'].unique()
    data_tmp = data_long.groupby(
        'trace_ID').mean().query('2*std < accum').reset_index()
    IDs_keep = data_tmp['trace_ID']
    data_long = data_long[
        data_long['trace_ID'].isin(IDs_keep)]

    # Remove time series with fewer than 5 years
    data_tmp = data_long.join(
        data_long.groupby('trace_ID')['Year'].count(), 
        on='trace_ID', rsuffix='_count')
    data_long = data_tmp.query('Year_count >= 5').drop(
        'Year_count', axis=1)

    # Reset trace IDs to match total number of traces
    tmp_group = data_long.groupby('trace_ID')
    data_long['trace_ID'] = tmp_group.ngroup()
    data_final = data_long.sort_values(
        ['trace_ID', 'Year']).reset_index(drop=True)

    return data_final

def long2gdf(accum_long):
    """
    Function to convert data in long format to geodataframe aggregated
    by trace location and sorted by collection time
    """

    accum_long['collect_time'] = (
        accum_long.collect_time.values.astype(np.int64))
    traces = accum_long.groupby('trace_ID').mean().drop(
        'Year', axis=1)
    traces['collect_time'] = (
        pd.to_datetime(traces.collect_time)
        .dt.round('1ms'))

    # Sort by collect_time and reset trace_ID index
    # traces = (traces.sort_values('collect_time')
    #     .reset_index(drop=True))
    # traces.index.name = 'trace_ID'
    traces = traces.reset_index()

    gdf_traces = gpd.GeoDataFrame(
        traces.drop(['Lat', 'Lon'], axis=1), 
        geometry=gpd.points_from_xy(
            traces.Lon, traces.Lat), 
        crs="EPSG:4326")

    return gdf_traces

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

    if k_neighbors==2:
        # Select 2nd closest (as first closest will be the same point)
        closest = indices[1]
        closest_dist = distances[1]
    else:
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

    end_crs = left_gdf.crs

    # Ensures data are in WGS84 projection
    left_gdf = left_gdf.to_crs('EPSG:4326')
    right_gdf = right_gdf.to_crs('EPSG:4326')

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

    closest, dist = get_nearest(
        src_points=left_radians, 
        candidates=right_radians)

    # Return points from right GeoDataFrame that are closest to points in left GeoDataFrame
    closest_points = right.loc[closest]

    # Ensure that the index corresponds the one in left_gdf
    closest_points = closest_points.reset_index(drop=True)

    # Add distance if requested
    if return_dist:
        # Convert to meters from radians
        closest_points['distance'] = dist * planet_radius

    # Reproject results back to original projection
    closest_points = closest_points.to_crs(end_crs)

    return closest_points


def get_Xovers(
    gdf_Xover, gdf_candidates, cutoff_dist):
    """
    Function to extract crossover locations (within a cutoff distance) and return traceID values at those locations. 

    NOTE: Assumes that input gdf is for Antarctica.
    """

    gdf_Xover = gdf_Xover.to_crs("EPSG:3031")
    gdf_candidates = gdf_candidates.to_crs(
        "EPSG:3031")
    
    gdf_Xover['geometry'] = gdf_Xover.buffer(
        cutoff_dist)

    candidate_near = gpd.sjoin(
        gdf_candidates, gdf_Xover).drop(
            columns=['index_right'])
    
    Xover_idx1 = candidate_near.collect_time.argmax()
    Xover_idx2 = candidate_near.collect_time.argmin()
    trace_ID1 = candidate_near['trace_ID'].iloc[
        Xover_idx1]
    trace_ID2 = candidate_near['trace_ID'].iloc[
        Xover_idx2]

    t_delta = (
        candidate_near.collect_time.max() 
        - candidate_near.collect_time.min()
    )
    print(f"Time difference between traces is {t_delta}")

    return trace_ID1, trace_ID2

def plot_Xover(
    accum_data, std_data, ts_trace1, ts_trace2):

    """
    Function to generate plot objects comparing the time series of two cross-over locations of the same flightline.
    """

    ts1 = pd.DataFrame(
        {'accum': accum_data[ts_trace1], 
        'upper': accum_data[ts_trace1] 
            + std_data[ts_trace1], 
        'lower': accum_data[ts_trace1] 
            - std_data[ts_trace1]})
    ts2 = pd.DataFrame(
        {'accum': accum_data[ts_trace2], 
        'upper': accum_data[ts_trace2] 
            + std_data[ts_trace2], 
        'lower': accum_data[ts_trace2] 
            - std_data[ts_trace2]})

    plt_ts1 = (
        hv.Curve(data=ts1, 
            kdims=['Year', 'accum']).opts(
            color='blue', line_width=2) 
        * hv.Curve(data=ts1, 
        kdims=['Year','upper']).opts(
            color='blue', line_dash='dotted') 
        * hv.Curve(data=ts1, 
            kdims=['Year','lower']).opts(
            color='blue', line_dash='dotted'))
    plt_ts2 = (
        hv.Curve(data=ts2, 
            kdims=['Year', 'accum']).opts(
            color='red', line_width=2) 
        * hv.Curve(data=ts2, 
        kdims=['Year','upper']).opts(
            color='red', line_dash='dotted') 
        * hv.Curve(data=ts2, 
            kdims=['Year','lower']).opts(
            color='red', line_dash='dotted'))
    
    return plt_ts1, plt_ts2

def trend_bs(df, nsim, df_err=pd.DataFrame()):
    """
    Dpc string goes here.
    """
    tic = time.perf_counter()

    trends_bs = pd.DataFrame(columns=df.columns)
    intercepts = pd.DataFrame(columns=df.columns)

    if df_err.empty:
        df_err = pd.DataFrame(
            np.ones(df.shape), index=df.index, 
            columns=df.columns)

    for _ in range(nsim):
        data_bs = df.sample(
        len(df), replace=True).sort_index()
        weights_bs = (
            1/df_err.loc[data_bs.index]).mean(axis=1)
        coeffs = np.polyfit(
            data_bs.index, data_bs, 1, w=weights_bs)

        trends_bs = trends_bs.append(
            pd.Series(coeffs[0], index=df.columns), 
            ignore_index=True)
        intercepts = intercepts.append(
            pd.Series(coeffs[1], index=df.columns), 
            ignore_index=True)

    toc = time.perf_counter()
    print(f"Execution time of bootstrapping: {toc-tic} s")

    trend_mu = np.mean(trends_bs)
    intercept_mu = np.mean(intercepts)
    trend_lb = np.percentile(trends_bs, 2.5, axis=0)
    trend_ub = np.percentile(trends_bs, 97.5, axis=0)

    return trend_mu, intercept_mu, trend_lb, trend_ub

# Function to perform autocorrelation
def acf(df):
    """
    Doc string goes here.
    """
    # Detrend time series data
    arr_ts = signal.detrend(df, axis=0)

    lags = int(np.round(arr_ts.shape[0]/2))
    arr_acf = np.zeros((lags, arr_ts.shape[1]))
    for idx, col in enumerate(arr_ts.T):
        arr_acf[:,idx] = tsa.acf(col, nlags=lags-1)
    acf_df = pd.DataFrame(
        arr_acf, columns=df.columns, 
        index=np.arange(lags))
    acf_df.index.name = 'Lag'

    return acf_df





    