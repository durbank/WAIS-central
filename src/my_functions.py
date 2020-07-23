# Module script to store project-specific custom functions and variables

# Import requisite modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import time
from sklearn.neighbors import BallTree
from scipy import signal
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

    # # Ensure each trace has only one time series 
    # # (if not, take the mean of all time series)
    # data = data.groupby(['trace_ID', 'Year']).mean()

    # Generate descriptive statistics based on imported 
    # gamma-fitted parameters
    alpha = data['gamma_shape']
    alpha.loc[alpha<1] = 1
    beta = 1/data['gamma_scale']
    mode_accum = (alpha-1)/beta
    var_accum = alpha/beta**2

    # New df (in long format) with accum data assigned
    data_long = (
        data.filter(['trace_ID', 'collect_time', 
        'QC_flag', 'Lat', 'Lon', 
        'elev', 'Year']).assign(
            accum=mode_accum, std=np.sqrt(var_accum))
        .reset_index()
    )
    data_long['collect_time'] = pd.to_datetime(
        data_long['collect_time'])
    return data_long

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

def trend_bs(df, nsim, weights=[]):
    """
    Dpc string goes here.
    """
    tic = time.perf_counter()

    trends_bs = pd.DataFrame(columns=df.columns)
    intercepts = pd.DataFrame(columns=df.columns)

    if weights:
        weights.name = 'weights'

    for _ in range(nsim):
        data_bs = df.sample(
        len(df), replace=True).sort_index()
        # weights_bs = weights[accum_bs.index]
        # coeffs = np.polyfit(
        #     accum_bs.index, accum_bs, 1, w=1/weights_bs)
        coeffs = np.polyfit(data_bs.index, data_bs, 1)
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
def acf(series):
    data = signal.detrend(series)
    n = len(data)
    variance = data.var()
    x = data-data.mean()
    r = np.correlate(x, x, mode = 'same')
    result = r/(variance*n)
    return result

# Function to perform spectral analysis
def get_spectrum(df, coeff1, coeff0):
    """
    Doc string goes here.
    """
    # Detrend data (may need to use coeffs here for speed)
    data = signal.detrend(df)


    