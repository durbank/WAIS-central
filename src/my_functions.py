# Module script to store project-specific custom functions and variables

# Import requisite modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import time
from sklearn.neighbors import BallTree

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






def trend_bs(df, nsim, weights=[]):



    tic = time.perf_counter()

    trends_bs = pd.DataFrame(columns=df.columns)

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

    toc = time.perf_counter()
    print(f"Execution time of bootstrapping: {toc-tic} s")

    trend_mu = np.mean(trends_bs)
    trend_lb = np.percentile(trends_bs, 2.5, axis=0)
    trend_ub = np.percentile(trends_bs, 97.5, axis=0)

    return trend_mu, trend_lb, trend_ub