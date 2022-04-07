# Script for preprocessing WAIS data in preparation to analysis in R using INLA.

## Imports
from pathlib import Path
from importlib import util
import pandas as pd

# Define environment directories
ROOT_DIR = Path().absolute().parents[0]
EXT_DIR = Path('/home/durbank/Documents/Research/Antarctica/WAIS-central/')

# Import custom paipr processing module
spec = util.spec_from_file_location(
    "paipr", EXT_DIR.joinpath('src/my_mods/paipr.py'))
paipr = util.module_from_spec(spec)
spec.loader.exec_module(paipr)

# Import raw data
data_list = [folder for folder in EXT_DIR.joinpath(
    'data/PAIPR-outputs').glob('*')]
data_raw = pd.DataFrame()
for folder in data_list:
    data = paipr.import_PAIPR(folder)
    data_raw = data_raw.append(data)

# Remove results for below QC data reliability
data_raw.query('QC_flag != 2', inplace=True)
data_0 = data_raw.query(
    'Year > QC_yr').sort_values(
    ['collect_time', 'Year']).reset_index(drop=True)

# Format and sort results for further processing
data_form = paipr.format_PAIPR(data_0)

# Create time series arrays for annual accumulation 
# and error
data_all = data_form.drop(['QC_flag', 'QC_med'], axis=1)
accum_ALL = data_form.pivot(
    index='Year', columns='trace_ID', values='accum')
std_ALL = data_form.pivot(
    index='Year', columns='trace_ID', values='std')

# Create gdf of mean results for each trace and 
# transform to Antarctic Polar Stereographic
gdf_traces = paipr.long2gdf(data_form)
gdf_traces.to_crs(epsg=3031, inplace=True)

# Save data to disk
data_all.to_csv(ROOT_DIR.joinpath('data/paipr-long.csv'))
accum_ALL.to_csv(ROOT_DIR.joinpath('data/paipr-accum.csv'))
std_ALL.to_csv(ROOT_DIR.joinpath('data/paipr-std.csv'))
gdf_traces.to_file(ROOT_DIR.joinpath('data/traces.geojson'), driver='GeoJSON')
