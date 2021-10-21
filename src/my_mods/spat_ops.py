# Module for various spatial manipulations and extractions of both raster and vector data

# Import required modules
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.neighbors import BallTree
import xarray as xr

def pts2grid(geo_df, resolution=5000):
    import shapely
    """
    Description.
    """
    xmin, ymin, xmax, ymax = geo_df.total_bounds

    x_pts = np.arange(
        np.floor(xmin), np.ceil(xmax)+resolution, 
        step=resolution)
    y_pts = np.arange(
        np.floor(ymin), np.ceil(ymax)+resolution, 
        step=resolution)

    grid_cells = []
    for x in x_pts:
        for y in y_pts:
            x0 = x - resolution
            y0 = y - resolution
            grid_cells.append(shapely.geometry.box(x0, y0, x, y))
    
    geo_cells = gpd.GeoDataFrame(
        data=grid_cells, columns=['geometry'],
        crs=geo_df.crs)
    # # Diagnostic plot
    # ax = geo_cells.plot(facecolor='none', edgecolor='grey')
    # geo_df.plot(ax=ax, column='accum', cmap='viridis', vmax=600)

    geo_sjoin = gpd.sjoin(
        geo_df, geo_cells, how='inner', op='within')
    gdf_grid = gpd.GeoDataFrame(
        data=geo_sjoin[['trace_ID', 'elev', 'accum', 'std', 'index_right']], 
        geometry=geo_cells.geometry[geo_sjoin['index_right']].values, 
        crs=geo_df.crs)
    
    return gdf_grid


def trace_combine(gdf_grid, accum_ALL, std_ALL):
    """
    Description.
    """
    # Get indices of grids that are populated
    grid_pop = np.unique(gdf_grid['index_right'])

    # Preallocate return arrays
    accum_df = pd.DataFrame(
        columns=np.arange(len(grid_pop)), 
        index=accum_ALL.index)
    MoE_df = pd.DataFrame(
        columns=np.arange(len(grid_pop)), 
        index=accum_ALL.index)
    yr_count = pd.DataFrame(
        columns=np.arange(len(grid_pop)), 
        index=accum_ALL.index)
    grid_final = gpd.GeoDataFrame(
        columns=[
            'grid_ID', 'elev', 'trace_count', 
            'accum', 'MoE', 'geometry'], 
        index=np.arange(len(grid_pop)), crs=gdf_grid.crs)
    

    for i, grid_idx in enumerate(grid_pop):

        # Subset geodf to traces in current grid and get trace_IDs
        gdf_tmp = gdf_grid[gdf_grid['index_right']==grid_idx]
        t_IDs = gdf_tmp['trace_ID'].values

        # Get subsetted accum and std arrays
        accum_arr = accum_ALL.iloc[:,t_IDs]
        std_arr = std_ALL.iloc[:,t_IDs]

        nsim = 10
        accum_sim = pd.DataFrame(np.zeros(
            (accum_arr.shape[0], nsim*accum_arr.shape[1])), 
            index=accum_arr.index)
        for j in range(accum_arr.shape[1]):

            col_mu = accum_arr.iloc[:,j]
            col_std = std_arr.iloc[:,j]

            for k in range(nsim):
                accum_sim.iloc[:,(j*nsim+k)] = np.random.normal(
                    loc=col_mu, scale=col_std)

        accum_mu = accum_sim.mean(axis=1)
        n_count = accum_sim.count(axis=1)
        accum_moe = (
            1.96*accum_sim.std(axis=1) 
            / np.sqrt(n_count))

        # # Calculate weighted mean accum and std
        # weights = (1/std_arr) / np.tile(
        #     (1/std_arr).sum(axis=1).to_numpy(), 
        #     (std_arr.shape[1],1)). transpose()
        # accum_w = (accum_arr*weights).sum(axis=1)
        # std_w = np.sqrt(((std_arr**2)*weights).sum(axis=1))
        # accum_mu = accum_w
        # accum_moe = (
        #     1.96*std_w
        #     / np.sqrt(accum_arr.count(axis=1)))

        # Set missing data to NaN
        nan_idx = np.invert(
            accum_arr.sum(axis=1).astype('bool'))
        accum_mu[nan_idx] = np.nan
        accum_moe[nan_idx] = np.nan

        # Add weighted mean and std to return arrays
        accum_df.iloc[:,i] = accum_mu
        MoE_df.iloc[:,i] = accum_moe

        # Get counts for number of records in annual 
        # estimates and add to return array
        tmp_yr = accum_arr.count(axis=1)
        tmp_yr[nan_idx] = np.nan
        yr_count.iloc[:,i] = tmp_yr

        # Populate return geodf
        grid_final.loc[i,'grid_ID'] = grid_idx
        grid_final.loc[i,'elev'] = gdf_tmp['elev'].mean()
        grid_final.loc[i, 'trace_count'] = len(t_IDs)
        grid_final.loc[i,'accum'] = accum_mu.mean()
        grid_final.loc[i,'MoE'] = (
            1.96*accum_mu.std()/np.sqrt(accum_mu.count()))
        grid_final.loc[i,'geometry'] = gdf_tmp.iloc[0].geometry

    return grid_final, accum_df, MoE_df, yr_count

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

def extract_at_pts(
    xr_ds, gdf_pts, coord_names=['lon','lat'], 
    return_dist=False, planet_radius=6371000):
    """
    Function where, given an xr-dataset and a Point-based geodataframe, extract all values of variables in xr-dataset at pixels nearest the given points in the geodataframe.
    xr_ds {xarray.core.dataset.Dataset}: Xarray dataset containing variables to extract.
    gdf_pts {geopandas.geodataframe.GeoDataFrame} : A Points-based geodataframe containing the locations at which to extract xrarray variables.
    coord_names {list}: The names of the longitude and latitude coordinates within xr_ds.
    return_dist {bool}: Whether function to append the distance (in meters) between the given queried points and the nearest raster pixel centroids. 
    NOTE: This assumes the xr-dataset includes lon/lat in the coordinates 
    (although they can be named anything, as this can be prescribed in the `coord_names` variable).
    """

    # Convert xr dataset to df and extract coordinates
    xr_df = xr_ds.to_dataframe().reset_index()
    xr_coord = xr_df[coord_names]

    # Ensure gdf_pts is in lon/lat and extract coordinates
    crs_end = gdf_pts.crs 
    gdf_pts.to_crs(epsg=4326, inplace=True)
    pt_coord = pd.DataFrame(
        {'Lon': gdf_pts.geometry.x, 
        'Lat': gdf_pts.geometry.y}).reset_index(drop=True)

    # Convert lon/lat points to RADIANS for both datasets
    xr_coord = xr_coord*np.pi/180
    pt_coord = pt_coord*np.pi/180

    # Find xr data nearest given points
    xr_idx, xr_dist = get_nearest(pt_coord, xr_coord)

    # Drop coordinate data from xr (leaves raster values)
    cols_drop = list(dict(xr_ds.coords).keys())
    xr_df_filt = xr_df.iloc[xr_idx].drop(
        cols_drop, axis=1).reset_index(drop=True)
    
    # Add raster values to geodf
    gdf_return = gdf_pts.reset_index(
        drop=True).join(xr_df_filt)
    
    # Add distance between raster center and points to gdf
    if return_dist:
        gdf_return['dist_m'] = xr_dist * planet_radius
    
    # Reproject results back to original projection
    gdf_return.to_crs(crs_end, inplace=True)

    return gdf_return

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



# Experiments with unit testing (currently does nothing)
if (__name__ == '__main__'):
    # import sys
    print('YAY FOR UNIT TESTING')