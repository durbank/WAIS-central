# Script to map out the locations of cores used for PAIPR

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
ROOT_DIR = Path(__file__).parents[1]

# Set project data directory
DATA_DIR = ROOT_DIR.joinpath('data')

# Define plotting projection to use
ANT_proj = ccrs.SouthPolarStereo(true_scale_latitude=-71)

# Import Antarctic outline shapefile
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ant_poly = world.query('continent == "Antarctica"')
ant_poly.to_crs(epsg=3031, inplace=True)
# ant_path = ROOT_DIR.joinpath(
#     'data/Ant_basemap/Coastline_medium_res_polygon.shp')
# ant_outline = gpd.read_file(ant_path)


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

samba_locs = gpd.GeoDataFrame(
    data=core_meta.drop(['Lat', 'Lon'], axis=1), 
    geometry=gpd.points_from_xy(
        core_meta.Lon, core_meta.Lat), 
    crs='EPSG:4326')
samba_locs.to_crs('EPSG:3031', inplace=True)



# Import SUMup dataset
core_file = Path(
    "/home/durbank/Documents/Durban/Research/Mentoring/"
    + "Laurie/Data/SUMup_datasets_july2018_density.csv")
cores_raw = pd.read_csv(core_file)
cores_raw = cores_raw[cores_raw['lat'] < -60][
    ['date', 'lat', 'lon', 'midpoint_depth', 
    'density', 'error', 'elevation']]
cores_raw = cores_raw.query('lat != -9999')
core_groups = cores_raw.groupby(['lat', 'lon', 'date'])

SUMup = core_groups.mean().reset_index()
SUMup_locs = gpd.GeoDataFrame(
    data=SUMup.drop(['lat','lon'], axis=1), 
    geometry=gpd.points_from_xy(
        SUMup.lon, SUMup.lat), 
    crs='EPSG:4326')
SUMup_locs.to_crs(epsg=3031, inplace=True)



ant_bnds = gv.Polygons(
    ant_poly, crs=ANT_proj).opts(
    projection=ANT_proj, color='grey', alpha=0.7)
samba_plt = gv.Points(
    data=samba_locs.query('Duration >= 10'), 
    crs=ANT_proj, vdims=['Name']).opts(
    projection=ANT_proj, color='blue', tools=['hover'])
SUMup_plt = gv.Points(
    data=SUMup_locs.query('midpoint_depth >= 7'), 
    crs=ANT_proj, vdims=['midpoint_depth']).opts(
    projection=ANT_proj, color='red', tools=['hover'])
(ant_bnds * samba_plt * SUMup_plt).opts(height=500, width=600)
(ant_bnds * SUMup_plt * samba_plt).opts(height=500, width=600)