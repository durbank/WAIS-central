# Script to generate data for use in notebooks

# Attach required packages
library(tidyverse)
library(here)
library(sf)
library(tcltk)
library(readxl)
library(broom)
library(rgdal)
library(raster)
library(tictoc)

tic("Total runtime")

##----
## Import and clean data

# Function to import all .csv files of long format PAIPR results from a given directory, and
# concatenate the results into a single tibble
import_PAIPR = function(input_dir) {
  files = list.files(input_dir, pattern = "*.csv", full.names = TRUE)
  data = tibble()
  for (f in files) {
    data_f = read_csv(f)
    data = rbind(data, data_f)
  }
  return(data)
}

# Import and format PAIPR-generated SMB results
data = import_PAIPR( here("data/gamma_20111109"))
data$Year = as.integer(data$Year)

# Generate descriptive statistics based on imported gamma-fitted parameters
alpha = data$gamma_shape
alpha[which(alpha<1)] = 1
beta = 1/data$gamma_scale
mode.accum = (alpha-1)/beta
var.accum = alpha/beta^2

# Make df of relevant data in long format
accum.long = data %>% dplyr::select(Lat, Lon, elev, Year) %>% 
  mutate(accum = mode.accum, sd = sqrt(var.accum))

# Find duplicate annual accum records and average into single record
accum.long = accum.long %>% group_by(Lat, Lon, elev, Year) %>%
  summarise(accum = mean(accum, na.rm = TRUE), sd = mean(sd, na.rm = TRUE)) %>% ungroup()

# This removes years prior to a cutoff, and clips the results 
# to only time series that extend to cutoff
yr.first = 1985
yr.last = 2007
accum.long = accum.long %>% filter(Year>=yr.first) %>% filter(Year <= yr.last)
data.cutoff = accum.long %>% group_by(Lat, Lon) %>% summarise(yr.start = min(Year))

# Nest year/accum data into list-columns for each location  
# (insert # prior to "filter" pipe to avoid cutoff)
accum.loc = nest(accum.long, data = c(Year, accum, sd)) %>% filter(data.cutoff$yr.start <= yr.first)
rm(data, data.cutoff, alpha, beta, mode.accum, var.accum)



##----
## Import core data and convert both datasets to sf geometry objects

# Create sf point object, where each row has a unique XYZ coordinate in space, 
# with associated list-columns for the geometry and data variables
accum.sf = st_as_sf(accum.loc, coords = c("Lon", "Lat"), dim = "XY")

# Define the coordinate reference system used in original data
st_crs(accum.sf) = "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84"

# Import basin shapefiles (in Antarctic Polar Sterographic EPSG:3031)
basins = st_read(here("data/ANT_Basins_IMBIE2_v1.6/ANT_Basins_IMBIE2_v1.6.shp"))

# Transform SMB sf object into APS projection
accum.PS = accum.sf %>% st_transform(st_crs(basins))
rm(accum.sf)

# Import Antarctic outline
Ant.outline = st_read(here("data/Ant_basemap/Coastline_medium_res_polygon.shp"))

# Import the SAMBA database from .xlsx file
samba = read_excel(here("data/DGK_SMB_compilation.xlsx"), 
                   sheet = "Accumulation", na = c("", "NAN", "NaN"))

# Remove columns with  no data
keep.idx = function(x) any(!is.na(x))
samba = samba %>% select_if(keep.idx)

# Create tbl of Site locations
core.locs = tibble(Site = colnames(samba[,2:ncol(samba)]), 
                   Latitude = as.numeric(samba[1,2:ncol(samba)]), 
                   Longitude = as.numeric(samba[2,2:ncol(samba)]), 
                   Elevation = as.numeric(samba[3,2:ncol(samba)]))

# Convert samba database to long format and rm NA values
core_long = samba[4:nrow(samba),] %>% rename(Year = Name) %>% 
  pivot_longer(-Year, names_to = "Site", values_to = "Accum") %>% 
  filter(!is.na(Accum))

# Recombine accum data in long format with location data and rearrange
cores.accum = left_join(core_long, core.locs, by = "Site")  %>% arrange(Site, Year) %>% 
  dplyr::select(Site, Latitude, Longitude, Elevation, Year, Accum) %>% 
  mutate(Year = as.integer(Year))

# Keep only accumulation years within specified range
cores.accum = cores.accum %>% filter(Year >= yr.first) %>% filter(Year <= yr.last)

# This clips the results to only time series that extend to a selected year
cores.cutoff = cores.accum %>% group_by(Site) %>% 
  summarise(yr.start = min(Year))

# Nest year/accum data into list-columns for each location  
# (insert # prior to "filter" pipe to avoid cutoff)
tmp.nest = nest(cores.accum, Data = c(Year, Accum)) %>% filter(cores.cutoff$yr.start <= yr.first)

# Add column for the length of Data records and remove cores shorter than 75% of given duration
cores.nest = mutate(tmp.nest, data.length = unlist(lapply(tmp.nest$Data, nrow))) %>% 
  filter(data.length >= (0.75*(yr.last-yr.first)))

# Convert to simple feature object
cores.PS = cores.nest %>% 
  st_as_sf(coords = c("Longitude", "Latitude"), dim = "XY", 
           crs = "+proj=longlat +datum=WGS84 +no_defs +ellps=WGS84") %>%
  st_transform(st_crs(Ant.outline))
rm(samba, keep.idx, core_long, tmp.nest)

# Crop cores to chosen spatial extent
cores.PS = cores.PS %>% st_crop(xmin=-1.5e6, xmax=-8e5, ymin=-7e5, ymax=1e5)



##----
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
dem.index = st_read(index.file)
# ggplot() + geom_sf(data = Ant.outline, color = 'grey') + geom_sf(data = dem.index)
keep.idx = unique(unlist(st_intersects(accum.PS, dem.index)))
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



##----
## Generate aspect and slope tiffs for any required REMA tiles, extract slope/aspect
## values for all accum points, and add data to data tibbles

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



##----
## Calculate mean accumulation rate for each point and generate non-spatially-
## referenced tibbles for accum and cores (makes some plotting/analysis easier)

# Find the mean accumulation at each data location across all years (this method strips NA 
# values first, so the exact temporal range differs by location)
accum.PS = accum.PS %>% 
  mutate(accum_mu = simplify(map(data, ~ mean(.x$accum, na.rm = TRUE))))

# # Create non-referenced tibble of accumulation by location
# accum.loc_PS = as_tibble(st_coordinates(accum.PS)) %>%
#   rename(Easting = X, Northing = Y) %>% bind_cols(st_set_geometry(accum.PS, NULL))

# Find the mean accumulation at each core location across all years (this method strips NA 
# values first, so the exact temporal range differs by location)
cores.PS = cores.PS %>% 
  mutate(accum_mu = simplify(map(Data, ~ mean(.x$Accum, na.rm = TRUE))))

# # Create non-referenced tibble of core accumulation by location
# cores.loc_PS = as_tibble(st_coordinates(cores.PS)) %>%
#   rename(Easting = X, Northing = Y) %>% bind_cols(st_set_geometry(cores.PS, NULL))



##----
## Bootstrapping tests for time series regression

# library(car)
# test = accum.loc_PS %>% sample_n(size=1000)
# test$trend = NA
# test$bias = NA
# test$St.err = NA
# test$conf2.5 = NA
# test$conf97.5 = NA
# # test$r2 = NA
# tic("Bootstrap timer:")
# for (i in 1:nrow(test)) {
#   data.i = test$data[[i]]
#   lm.i = lm(accum ~ Year, data = data.i, weights = 1/sd^2)
#   # bs = function(formula, data, indices) {
#   #   data_smp = data[indices,]
#   #   fit = lm(accum ~ Year, data=data_smp, weights = 1/sd^2)
#   #   return(coef(fit))
#   # }
#   # boot.i = boot(data=data.i, statistic=bs, R=1000)
#   boot.i = Boot(lm.i, R=200, ncores = 4)
#   boot.sum = summary(boot.i)
#   boot.conf = confint(boot.i)
#   
#   test$trend[i] = boot.sum$original[2]
#   test$bias[i] = boot.sum$bootBias[2]
#   test$St.err[i] = boot.sum$bootSE[2]
#   test$conf2.5[i] = boot.conf[2,1]
#   test$conf97.5[i] = boot.conf[2,2]
# }
# toc()
# 
# # test.lm = lm(accum ~ Year, data = test$data[[1]], weights = 1/sd^2)
# # test.boot = Boot(test.lm)
# # summary(test.boot)
# # 
# # test.lm = map(test$data, ~ lm(accum ~ Year, data = .x, weights = 1/sd^2))
# # 
# # boot.fun = function(x)  Boot(x, R=500)
# # test.boot = lapply(test.lm, boot.fun)
# # 
# # test.boot = test.lm %>% map(function(x) Boot(x, R=500))



##----
## Accumulation time series regression

# Create linear models for time series at each location in data
accum.lm = map(accum.PS$data, ~ lm(accum ~ Year, data = .x, weights = 1/sd^2))

# I should correct significance values based on autocorrelation in the data
# It would be interesting to investigate the 2nd derivative in the data 
# (the trend through time of the trend). 
# One approach to this would be to fit a line through each time series, but iteratively 
# shorten the time series. In this way, I can see if the trend changes (e.g. accelerates) 
# during recent years.
# Another approach could be to perform linear fits of each time series over a ?10?-year moving window

# # Extract relevant model data and store in nested tibble
# lm.accum_data = dplyr::select(accum.PS, -data, -accum_mu) %>% 
#   mutate(data = map(accum.lm, 
#                     ~ dplyr::select(augment(.), Year, accum, .fitted, .se.fit, .resid, .std.resid)))

# Extract model parameter estimates and p values and store as location-based tibble
tmp.tidy = map(accum.lm, ~ tidy(.x))
accum.PS = accum.PS %>% 
  mutate(intrcpt = simplify(map(tmp.tidy, "estimate") %>% map(1))) %>%
  mutate(coeff_yr = simplify(map(tmp.tidy, "estimate") %>% map(2))) %>% 
  mutate(std.err = simplify(map(tmp.tidy, "std.error") %>% map(2))) %>%
  mutate(p.val_yr = simplify(map(tmp.tidy, "p.value") %>% map(2))) %>% 
  mutate(coeff_perc = coeff_yr/accum_mu, err_perc = std.err/accum_mu)
# trend.loc = accum.loc_PS %>% 
#   dplyr::select(Easting, Northing, elev, accum_mu) %>% 
#   mutate(intrcpt = simplify(map(tmp.tidy, "estimate") %>% map(1))) %>%
#   mutate(coeff_yr = simplify(map(tmp.tidy, "estimate") %>% map(2))) %>% 
#   mutate(std.err = simplify(map(tmp.tidy, "std.error") %>% map(2))) %>% 
#   mutate(p.val_yr = simplify(map(tmp.tidy, "p.value") %>% map(2)))

# # Scale trend values by the mean in-situ accumulation rate
# trend.scaled = trend.loc %>% mutate(coeff_yr = coeff_yr/accum_mu, 
#                                     std.err = std.err/accum_mu)

##### Same thing as above, but for the cores

# Create linear models for time series at each location in data
cores.lm = map(cores.PS$Data, ~ lm(Accum ~ Year, data = .x))

# Extract relevant model data and store in nested tibble
lm.core_data = dplyr::select(cores.PS, -Data, -accum_mu, -data.length) %>% 
  mutate(data = map(cores.lm, 
                    ~ dplyr::select(augment(.), Year, Accum, .fitted, .se.fit, .resid, .std.resid)))

# Extract model parameter estimates and p values and store as location-based tibble
tmp.tidy = map(cores.lm, ~ tidy(.x))
cores.PS = cores.PS %>% 
  mutate(intrcpt = simplify(map(tmp.tidy, "estimate") %>% map(1))) %>%
  mutate(coeff_yr = simplify(map(tmp.tidy, "estimate") %>% map(2))) %>% 
  mutate(std.err = simplify(map(tmp.tidy, "std.error") %>% map(2))) %>% 
  mutate(p.val_yr = simplify(map(tmp.tidy, "p.value") %>% map(2))) %>% 
  mutate(coeff_perc = coeff_yr/accum_mu, err_perc = std.err/accum_mu)

# # Scale trend values by the mean in-situ accumulation rate
# cores.scaled = cores.trend %>% mutate(coeff_yr = coeff_yr/accum_mu, 
#                                       std.err = std.err/accum_mu)

write_rds(accum.PS, here('data/interim_results/accum_data.rds'))
write_rds(cores.PS, here("data/interim_results/core_data.rds"))

toc()