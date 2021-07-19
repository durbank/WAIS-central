# Module containing custon stats functions and operations

# Requisite modules
import pandas as pd
import time
import numpy as np
from scipy import signal
import statsmodels.tsa.stattools as tsa
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde as kde

def trend_bs(
    df, nsim, df_err=pd.DataFrame(), 
    pval=False, n_samples=None):
    """
    Dpc string goes here.
    """
    tic = time.perf_counter()
    
    # Preallocate results arrays
    trends_bs = pd.DataFrame(columns=df.columns)
    intercepts = pd.DataFrame(columns=df.columns)

    # If no errors exist, create neutral weights
    if df_err.empty:
        df_err = pd.DataFrame(
            np.ones(df.shape), index=df.index, 
            columns=df.columns)

    # Peform bootstrapping
    for _ in range(nsim):
        # Randomly resample data
        data_bs = df.sample(
            len(df), replace=True).sort_index()
        
        # Generate mean weights
        weights_bs = (
                1/df_err.loc[data_bs.index]).mean(axis=1)

        # Check for nans (requires for loop w/ 2D array)
        if np.any(np.isnan(data_bs)):
            
            data_ma = np.ma.array(data_bs, mask=np.isnan(data_bs))
            coeffs = np.zeros((2,data_ma.shape[1]))
            for idx in range(coeffs.shape[1]):
                data_i = data_ma[:,idx]
                if (np.invert(data_i.mask)).sum() < 5:
                    coeffs[:,idx] = np.nan
                else:
                    coeffs[:,idx] = np.ma.polyfit(
                        data_bs.index, data_i, 1, w=weights_bs)
                
        
        else:
            # If no nan's present, perform vectorized fitting
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

    trend_mu = np.nanmean(trends_bs, axis=0)
    intercept_mu = np.nanmean(intercepts, axis=0)
    trendCI_lb = np.nanpercentile(trends_bs, 2.5, axis=0)
    trendCI_ub = np.nanpercentile(trends_bs, 97.5, axis=0)

    if pval:

        tic = time.perf_counter()

        # Generate null data
        df_null = pd.DataFrame(
            signal.detrend(df.fillna(df.mean()), axis=0), 
            index=df.index)
        
        null_trends = pd.DataFrame(columns=df_null.columns)

        # Generate null model estimates
        for _ in range(nsim):

            null_bs = df_null.sample(
                len(df_null), replace=True).sort_index()
            coeffs_null = np.polyfit(
                null_bs.index, null_bs, 1)
            null_trends = null_trends.append(
                pd.Series(coeffs_null[0], 
                index=df_null.columns), 
                ignore_index=True)
        
        
        if n_samples is None:
            n_samples = nsim

        # Estimate p-values based on comparison to null models
        pvals = np.zeros(len(trend_mu))
        for i, mu in enumerate(trend_mu):

            null_vals = null_trends.iloc[:,i]

            # Approximate continuous distribution using kde
            null_dist = kde(null_vals)
            null_samples = null_dist.resample(n_samples).squeeze()

            pvals[i] = (abs(null_samples) >= abs(mu)).sum() \
                / len(null_samples)

        toc = time.perf_counter()
        print(f"Execution time for p-values: {toc-tic} s")

        return trend_mu, intercept_mu, trendCI_lb, trendCI_ub, pvals

    else:    
        return trend_mu, intercept_mu, trendCI_lb, trendCI_ub

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

def vario(
    points_gdf, lag_size, 
    d_metric='euclidean', vars='all', 
    stationarize=False, scale=True):
    """A function to calculate the experimental variogram for values associated with a geoDataFrame of points.

    Args:
        points_gdf (geopandas.geodataframe.GeoDataFrame): Location of points and values associated with those points to use for calculating distances and lagged semivariance.
        lag_size (int): The size of the lagging interval to use for binning semivariance values.
        d_metric (str, optional): The distance metric to use when calculating pairwise distances in the geoDataFrame. Defaults to 'euclidean'.
        vars (list of str, optional): The names of variables to use when calculating semivariance. Defaults to 'all'.
        scale (bool, optional): Whether to perform normalization (relative to max value) on semivariance values. Defaults to True.

    Returns:
        pandas.core.frame.DataFrame: The calculated semivariance values for each chosen input. Also includes the lag interval (index), the average separation distance (dist), and the number of paired points within each interval (cnt). 
    """

    # Get column names if 'all' is selected for "vars"
    if vars == "all":
        vars = points_gdf.drop(
            columns='geometry').columns

    # Extact trace coordinates and calculate pairwise distance
    locs_arr = np.array(
        [points_gdf.geometry.x, points_gdf.geometry.y]).T
    dist_arr = pdist(locs_arr, metric=d_metric)

    # Calculate the indices used for each pairwise calculation
    i_idx = np.empty(dist_arr.shape)
    j_idx = np.empty(dist_arr.shape)
    m = locs_arr.shape[0]
    for i in range(m):
        for j in range(m):
            if i < j < m:
                i_idx[m*i + j - ((i + 2)*(i + 1))//2] = i
                j_idx[m*i + j - ((i + 2)*(i + 1))//2] = j


    if stationarize:
        pass
    
    # Create dfs for paired-point values
    i_vals = points_gdf[vars].iloc[i_idx].reset_index(
        drop=True)
    j_vals = points_gdf[vars].iloc[j_idx].reset_index(
        drop=True)

    # Calculate squared difference bewteen variable values
    sqdiff_df = (i_vals - j_vals)**2
    sqdiff_df['dist'] = dist_arr

    # Create array of lag interval endpoints
    d_max = lag_size * (dist_arr.max() // lag_size + 1)
    lags = np.arange(0,d_max+1,lag_size)

    # Group variables based on lagged distance intervals
    df_groups = sqdiff_df.groupby(
        pd.cut(sqdiff_df['dist'], lags))

    # Calculate semivariance at each lag for each variable
    gamma_vals = (1/2)*df_groups[vars].mean()
    gamma_vals.index.name = 'lag'

    if scale:
        gamma_df = gamma_vals / gamma_vals.max()
    else:
        gamma_df = gamma_vals

    # Add distance, lag center, and count values to output
    gamma_df['dist'] = df_groups['dist'].mean()
    gamma_df['lag_cent'] = lags[1::]-lag_size//2
    gamma_df['cnt'] = df_groups['dist'].count()

    return gamma_df

# Experiments with unit testing (currently does nothing)
if (__name__ == '__main__'):
    # import sys
    print('YAY FOR UNIT TESTING')