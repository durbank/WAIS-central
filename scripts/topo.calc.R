# Script to perform raster calculations and extraction for REMA topographic values

# Import libraries
library(here)
library(tidyverse)
library(sf)
library(terra)

## Find which REMA DEM 8-meter tiles intersect our dataset and download any tiles not
## already present in the selected directory "output.dir"

# Select directory containing REMA mosaic tiles (based on currently-running OS)
os = Sys.info()['sysname']
if (os == 'Linux') {
  REMA.dir = file.path("/media/durbank/WARP/Research/Antarctica/Data/DEMs/REMA/tiles_8m_v1.1")
} else {
  REMA.dir = file.path("G:/Research/Antarctica/Data/DEMs/REMA/tiles_8m_v1.1")
}
# REMA.dir = tk_choose.dir(caption = "Select directory in which to save REMA mosaic tiles")

# Find all mosaics that intersect the accumulation dataset
index.file = file.path(dirname(REMA.dir), "REMA_Tile_Index_Rel1.1/REMA_Tile_Index_Rel1.1.shp")
dem.index = st_read(index.file) %>% st_transform(crs = st_crs(gdf_traces))
# ggplot() + geom_sf(data = Ant.outline, color = 'grey') + geom_sf(data = dem.index)
keep.idx = unique(unlist(st_intersects(gdf_traces, dem.index)))
mosaic.download = dem.index[keep.idx,] %>% mutate(fileurl = as.character(fileurl))

# Download files from REMA ftp server
for (file in 1:nrow(mosaic.download)) {
  tmp.file = tempfile()
  dir.name = basename(dirname(mosaic.download$fileurl[file]))

  # If not already downloaded, download tile and extract
  if (!dir.exists(file.path(REMA.dir, dir.name))) {
    download.file(mosaic.download$fileurl[file], tmp.file)
    untar(tmp.file, exdir = file.path(REMA.dir, dir.name))
  }
  unlink(tmp.file)
}


## Generate aspect and slope tiffs for any required REMA tiles, extract slope/aspect
## values for all accum points, and add data to data tibbles
# 
# Function to find which accum points are within a given REMA 8-meter mosaic tile 
# and extract elevation, slope, and aspect from the REMA raster data
get_topo_vals = function(tile, accum.pts, tile.dir) {

  # Logical to filter accum data to those members within the tile extent
  pts.log = st_intersects(accum.pts, tile, sparse = FALSE)[,1]
  accum.i = accum.pts %>% filter(pts.log)

  # Generate file paths to DEM, slope raster and aspect raster
  dem.path = file.path(tile.dir, as.character(tile$tile),
                       paste(as.character(tile$tile), "_8m_dem.tif", sep = ""))
  slope.path = file.path(tile.dir, as.character(tile$tile),
                         paste(as.character(tile$tile), "_8m_slope.tif", sep = ""))
  aspect.path = file.path(tile.dir, as.character(tile$tile),
                          paste(as.character(tile$tile), "_8m_aspect.tif", sep = ""))

  # Check if slope/aspect rasters exist. If they don't, calculate them from DEM
  # and save to REMA directory
  if (!file.exists(slope.path)) {
    tile.dem = raster(dem.path)
    tile.slope = terrain(tile.dem, opt='slope', unit='degrees')
    writeRaster(tile.slope, filename = slope.path, format = "GTiff")
    tile.aspect = terrain(tile.dem, opt='aspect', unit='degrees')
    writeRaster(tile.aspect, filename = aspect.path, format = "GTiff")
  }

  # Generate raster stack of DEM, slope, and aspect
  # and extract raster values at accum points
  tile.stack = stack(list(dem.path, slope.path, aspect.path))
  stack.vals = extract(tile.stack, accum.i)

  # Get index values of accum data subset within the global accum dataset
  pts.idx = (1:length(pts.log))[pts.log]

  # Return index positions and raster stack values as list object
  return(list(pts.idx, stack.vals))
}

# Add variables for REMA elevation, slope, and aspect
accum.PS = accum.PS %>% mutate(elev.REMA = NA, slope = NA, aspect = NA)

for (i in 1:nrow(mosaic.download)) {

  # Get current tile with coverage in dataset
  tile = mosaic.download[i,]

  # Calculate and extract elevation, slope, and aspect
  vals.list = get_topo_vals(tile, accum.PS, REMA.dir)
  topo.idx = vals.list[[1]]
  topo.vals = vals.list[[2]]

  # Assign topo values to accum tbl
  accum.PS$elev.REMA[topo.idx] = topo.vals[,1]
  accum.PS$slope[topo.idx] = topo.vals[,2]
  accum.PS$aspect[topo.idx] = topo.vals[,3]
}

REMA.200m = "/media/durbank/WARP/Research/Antarctica/Data/DEMs/REMA/REMA_200m/REMA_200m_dem_filled.tif"
elev = rast(REMA.200m)
crs(r) = st_crs(gdf_traces)$wkt

slope.200m = "/media/durbank/WARP/Research/Antarctica/Data/DEMs/REMA/REMA_200m/REMA_200m_slope.tif"
slope = terrain(elev, v='slope', filename=slope.200m)

aspect.200m = "/media/durbank/WARP/Research/Antarctica/Data/DEMs/REMA/REMA_200m/REMA_200m_aspect.tif"
slope = terrain(elev, v='aspect', filename=aspect.200m)