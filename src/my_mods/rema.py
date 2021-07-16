# Module for downloading and manipulating REMA elevation data

# Requisite modules
from pathlib import Path
import requests
import shutil
import os
import richdem as rd
import numpy as np
import rasterio as rio

def get_REMA(tile_idx, output_dir):
    """
    Downloads, unzips, and saves DEM tiles from the Reference Elevation Model of Antarctica (REMA) dataset.
    Required inputs:
    tile_idx {pandas.core.frame.DataFrame}: Dataframe with the names (name), tile IDs (tile), and file urls (fileurl) of the requested data for download, taken from the [REMA tile shapefile](http://data.pgc.umn.edu/elev/dem/setsm/REMA/indexes/).
    output_dir {pathlib.PosixPath}: Path of directory at which to download the requested file.
    """

    for idx, row in tile_idx.iterrows():
        f_dir = output_dir.joinpath(row.tile)

        if not f_dir.exists():
            f_dir.mkdir(parents=True)
            zip_path = f_dir.joinpath('tmp.tar.gz')
            r = requests.get(row.fileurl, stream=True)
            print(f"Downloading tile {f_dir.name}")
            with open(zip_path, 'wb') as zFile:
                for chunk in r.iter_content(
                        chunk_size=1024*1024):
                    if chunk:
                        zFile.write(chunk)
            print(f"Unzipping tile {f_dir.name}")
            shutil.unpack_archive(zip_path, f_dir)
            os.remove(zip_path)
        else:
            print(f"REMA tile {f_dir.name} already exists locally, moving to next download")
    print("All requested files downloaded")


def calc_topo(dem_path):
    """
    Calculates slope and aspect from given DEM and saves output.
    The function checks to see whether a slope/aspect file has already been created so as to avoid needless processing.
    
    Parameters:
    dem_path (pathlib.PosixPath): The relative or absolute path to an input DEM file.

    Dependencies: 
    richdem module
    GDAL binaries
    pathlib module
    """
    slope_path = Path(
        str(dem_path).replace("dem", "slope"))
    aspect_path = Path(
        str(dem_path).replace("dem", "aspect"))

    if ((not slope_path.is_file()) or 
            (not aspect_path.is_file())):
        
        # Load DEM
        dem = rd.LoadGDAL(str(dem_path))
        
        # Calculate slope values from DEM
        if not slope_path.is_file():
            print(f"Calculating slope values for REMA tile {dem_path.parent.name}...")
            slope = rd.TerrainAttribute(
                dem, attrib='slope_riserun')
            rd.SaveGDAL(str(slope_path), slope)
        else:
                print(f"Slope data already exist locally for REMA tile {dem_path.parent.name}. Checking for aspect data...")
        
        # Calculate aspect values from DEM
        if not aspect_path.is_file():
            print(f"Calculating aspect values for REMA tile {dem_path.parent.name}...")
            aspect = rd.TerrainAttribute(dem, attrib='aspect')
            rd.SaveGDAL(str(aspect_path), aspect)
        else:
            print(f"Aspect data already exist locally for REMA tile {dem_path.parent.name}. Moving to next tile...")

    else:
        print(f"Slope/aspect geotifs already exist locally for REMA tile {dem_path.parent.name}. Moving to next tile...")


def topo_vals(tile_dir, locations, slope=False, aspect=False):
    """Extracts elevation, slope, and aspect values at given locations.
    Dependencies: Requires the rasterio (as rio) module and, by extension, GDAL binaries.
    Requires the geopandas module.

    Args:
        tile_dir (pathlib.PosixPath): The relative or absolute path to a directory containing REMA tile DSM, slope and aspect geotiffs.
        locations (geopandas.geodataframe.GeoDataFrame): A geodataframe containing the locations at which to extract raster data. These data should have the geometries stored in a column named "geometry" (the default for geopandas).
        slope (bool, optional): Whether to also extract slope values at points of interest. Defaults to False.
        aspect (bool, optional): Whether to also extract aspect values at points of interest. Defaults to False.

    Returns:
        geopandas.geodataframe.GeoDataFrame: The same input geodataframe with topographic values appended.
    """

    # Create empty Elev column if missing in gdf
    if 'elev' not in locations.columns:
        elev = np.empty(locations.shape[0])
        elev[:] = np.NaN
        locations['elev'] = elev

    # Get initial gdf crs
    crs_init = locations.crs

    # Ensure locations are in same crs as REMA (EPSG:3031)
    locations.to_crs(epsg=3031, inplace=True)

    # Extract coordinates of sample points
    coords = (
        [(x,y) for x, y in zip(
            locations.geometry.x, locations.geometry.y)]
    )

    # Extract elevation values for all points within tile
    tile_path = [
        file for file in tile_dir.glob("*dem.tif")][0]
    src = rio.open(tile_path)
    tile_vals = np.asarray(
        [x[0] for x in src.sample(coords, masked=True)])
    tile_mask = ~np.isnan(tile_vals)
    locations.loc[tile_mask,'elev'] = tile_vals[tile_mask]
    src.close()

    # Force elevation data to numeric
    #

    if slope:
        # Create empty slope column if missing in gdf
        if 'slope' not in locations.columns:
            slope = np.empty(locations.shape[0])
            slope[:] = np.NaN
            locations['slope'] = slope

        # Extract slope values for all points within tile
        tile_path = [
            file for file in tile_dir.glob("*slope.tif")][0]
        src = rio.open(tile_path)
        tile_vals = np.asarray(
            [x[0] for x in src.sample(coords, masked=True)])
        tile_mask = ~np.isnan(tile_vals)
        locations.loc[tile_mask,'slope'] = tile_vals[tile_mask]
        src.close()

    if aspect:
        # Create empty aspect column if missing in gdf
        if 'aspect' not in locations.columns:
            aspect = np.empty(locations.shape[0])
            aspect[:] = np.NaN
            locations['aspect'] = aspect

        # Extract aspect values for all points within tile
        tile_path = [
            file for file in tile_dir.glob("*aspect.tif")][0]
        src = rio.open(tile_path)
        tile_vals = np.asarray(
            [x[0] for x in src.sample(coords, masked=True)])
        tile_mask = ~np.isnan(tile_vals)
        locations.loc[tile_mask,'aspect'] = tile_vals[tile_mask]
        src.close()

    # Convert gdf crs back to original
    locations.to_crs(crs_init, inplace=True)

    return locations

# Experiments with unit testing (currently does nothing)
if (__name__ == '__main__'):
    # import sys
    print('YAY FOR UNIT TESTING')