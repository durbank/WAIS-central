# Script to take python-preprocessed data (generated using `preprocess.py`) and perform further R processing.
# The main points are to add additional topographic and climate covariates to the dfs of interest.

# Import libraries
library(here)
library(tidyverse)
library(sf)
library(terra)

# Import python data
data_long = read_csv(here('data/paipr-long.csv')) %>% select(-...1) %>% 
  mutate(Year = as.integer(Year))
gdf_traces = st_read(here('data/traces.geojson')) %>% as_tibble() %>% st_as_sf()

# Remove points in far northern extent
accum_bnds = st_coordinates(gdf_traces)
gdf_traces = gdf_traces %>% st_crop(xmin=-1.5e6, xmax=max(accum_bnds[,1]),
                                    ymin=min(accum_bnds[,2]), ymax=max(accum_bnds[,2]))

# # Small-scale testing bounds
# gdf_traces = gdf_traces %>% st_crop(xmin=-1.4e6, xmax=-1.25e6,
#                                     ymin=min(accum_bnds[,2]), ymax=-3e5)

# Remove unrealistically high elevations and overly large accum sites
trace_drop = gdf_traces %>% filter(elev > 3000 | accum > 1000)

# Add East/North coordinates and drop filtered sites
pts_tmp = st_coordinates(gdf_traces)
coord_df = tibble(trace_ID = gdf_traces$trace_ID, East = pts_tmp[,1], North = pts_tmp[,2])
data = data_long %>% left_join(coord_df) %>% filter(!trace_ID %in% trace_drop$trace_ID)
gdf_traces = gdf_traces %>% filter(!trace_ID %in% trace_drop$trace_ID)

# Import REMA dem data
REMA.dir = file.path("/media/durbank/WARP/Research/Antarctica/Data/DEMs/REMA/REMA_200m")
dem = rast(file.path(REMA.dir, "REMA_200m_dem.tif"))
crs(dem) = st_crs(gdf_traces)$wkt

# Import REMA slope data (calculated using topo.calc script)
slp = rast(file.path(REMA.dir, "REMA_200m_slope.tif"))
crs(slp) = crs(dem)

# Import REMA aspect data (calculated using topo.calc script)
aspct = rast(file.path(REMA.dir, "REMA_200m_aspect.tif"))
crs(aspct) = crs(dem)

# Combine all topo data
topo.stk = c(dem, slp, aspct)

# Extract topo values at data locations
topo.pts = extract(topo.stk, vect(gdf_traces$geometry)) %>% as_tibble() %>% 
  rename(dem=REMA_200m_dem) %>% 
  mutate(trace.ID=gdf_traces$trace_ID, aspect=sin(pi/180*aspect))
topo.pts = topo.pts %>% select(-ID)

# Define RACMO data directory
RACMO.dir = file.path("/media/durbank/WARP/Research/Antarctica/Data", 
                      "/RACMO/2.3p2_yearly_ANT27_1979_2016/interim")
files = list.files(path = RACMO.dir, full.names = TRUE, 
                   pattern = "\\.tif$")

# Extract RACMO SMB at points
r.smb = rast(files[1])
names(r.smb) = 1979:2016
smb.pts = extract(r.smb, vect(gdf_traces$geometry)) %>% 
  as_tibble() %>% mutate(trace.ID=gdf_traces$trace_ID) %>% 
  pivot_longer(cols = num_range(prefix='', range=1979:2016), 
               names_to = 'Year', values_to = 'SMB') %>% 
  mutate(Year = as.integer(Year))

# Extract RACMO u10m wind data at points
r.u10m = rast(files[2])
names(r.u10m) = 1979:2016
u10m.pts = extract(r.u10m, vect(gdf_traces$geometry)) %>% 
  as_tibble() %>% 
  pivot_longer(cols = num_range(prefix='', range=1979:2016), 
               names_to = 'Year', values_to = 'u10m') %>% 
  mutate(Year = as.integer(Year))

# Extract RACMO v10m wind data at points
r.v10m = rast(files[3])
names(r.v10m) = 1979:2016
v10m.pts = extract(r.v10m, vect(gdf_traces$geometry)) %>% 
  as_tibble() %>% pivot_longer(cols = num_range(prefix='', range=1979:2016), 
                               names_to = 'Year', values_to = 'v10m') %>% 
  mutate(Year = as.integer(Year))

# Add RACMO data to df
racmo.pts = smb.pts %>% 
  mutate(u10m=u10m.pts$u10m, v10m=v10m.pts$v10m) %>% 
  select(-ID)
data = data %>% 
  left_join(racmo.pts, by = c("trace_ID" = "trace.ID", "Year")) %>% 
  left_join(topo.pts, by = c("trace_ID" = "trace.ID"))

# # Calculate wind and topo dot cross product
# data = data %>% mutate(S_x = abs(slope)*cos(asin(aspect)),
#                        S_y = abs(slope)*aspect) %>%
#   mutate(MSWD = S_x*u10m + S_y*v10m)

# Add columns for mean u10m and v10m speeds
mu.wind = data %>% group_by(trace_ID) %>% 
  summarize(mu.u10 = mean(u10m), mu.v10 = mean(v10m))
data = data %>% left_join(mu.wind, by = "trace_ID")

# Save results for later import and further modeling
saveRDS(gdf_traces, here('data/Rdata-gdf_trace.rds'))
saveRDS(data, here('data/Rdata-clean.rds'))