# Script to generate data and perform analyses for article on 
# Bayesian inference of WAIS-central accumulation trends

# Import libraries
library(here)
library(INLA)
library(dplyr)
library(tidyr)
library(broom)
library(terra)
library(spdep)

## DATA IMPORT AND PREPROCESSING

# Import raw(ish) data
data.all = readRDS(here('data/Rdata-clean.rds'))
gdf_traces = readRDS(here('data/Rdata-gdf_trace.rds'))

# # Clip bounds
# bbox = st_bbox(gdf_traces)
# clipper = c(
#   bbox[1]+0.1*(bbox[3]-bbox[1]),
#   bbox[2]+0.5*(bbox[4]-bbox[2]),
#   0.6*(bbox[3]-bbox[1])+bbox[1], 
#   bbox[4]
# )
# gdf_traces = st_crop(gdf_traces, clipper)

# Remove locations with less than 10 years of data
trace.keep = data.all %>% count(trace_ID) %>% filter(n>=10)
data.all = data.all %>% filter(trace_ID %in% trace.keep$trace_ID)
gdf_traces = gdf_traces %>% filter(trace_ID %in% trace.keep$trace_ID)

# Subset data to every 5th location (coarsens data to ~2 km)
skip.int = 10
gdf.idx = seq(1, nrow(gdf_traces), by=skip.int)
gdf_traces = gdf_traces[gdf.idx,]

# Filter data to subsetted period
data = data.all %>% filter(trace_ID %in% gdf_traces$trace_ID) %>%
  filter(Year >= 1975) %>%
  filter(Year < 2015) %>% arrange(trace_ID, Year)

# Remove gdf rows where all data have been filtered out
gdf_traces = gdf_traces %>% filter(trace_ID %in% data$trace_ID)

# Select variables of interest
dat = data %>% select(trace_ID, East, North, Year, accum, 
                      std, SMB, elev, dem, slope, aspect, 
                      # u10m, v10m, mu.u10, mu.v10
)

# Center covariates
dat = dat %>% 
  mutate(elev = elev-mean(elev, na.rm=TRUE), 
         dem=dem-mean(dem, na.rm=TRUE), slope=slope-mean(slope, na.rm=TRUE), 
         # u10m=u10m-mean(u10m, na.rm=TRUE), v10m=v10m-mean(v10m, na.rm=TRUE), 
         # mu.u10=mu.u10-mean(mu.u10, na.rm=TRUE), mu.v10=mu.v10-mean(mu.v10, na.rm=TRUE)
  ) %>% 
  mutate(Year.mod = Year-mean(1975:2014), Year.idx = (Year-min(Year)+1))
# u10m=mean(u10m, na.rm=TRUE), v10m=mean(v10m, na.rm=TRUE))

# Scale covariates by sd
dat = dat %>% 
  mutate(elev = elev/sd(elev), dem=dem/sd(dem), 
         slope=slope/sd(slope), 
         # u10m=u10m/sd(u10m), v10m=v10m/sd(v10m), mu.u10=sd(mu.u10), mu.v10=sd(mu.v10)
  )

# Split into training and testing sets
dat = dat %>% mutate(row.ID = 1:nrow(dat)) %>% relocate(row.ID)
dat.train = dat %>% slice_sample(prop = 0.80) %>% arrange(row.ID)
dat.test = dat %>% filter(!row.ID %in% dat.train$row.ID)

# Directly calculate linear coefficients of time from raw data.
yr.trends = dat.train %>% 
  group_by(trace_ID) %>% 
  do(tidy(lm(accum ~ Year.mod, data = .))) %>% 
  filter(term=='Year.mod') %>% select(-term)
dat.mu = dat.train %>% group_by(trace_ID) %>% 
  summarize(East=mean(East), North=mean(North), accum=mean(accum)) %>% 
  left_join(yr.trends %>% select(trace_ID, estimate)) %>% 
  mutate(log.est = log(1+(estimate/accum)))

## INLA MODELING

# Mesh grid creation
# mesh = inla.mesh.2d(loc = dat.train %>% select(East, North),
#                     max.edge = c(17000, 100000), cutoff = 500)
mesh = inla.mesh.2d(loc = dat.train %>% select(East, North),
                    max.edge = c(30000, 100000), cutoff = 5000)
# plot(mesh)
# points(dat.train %>% select(East, North), col = "red")

# 2D spatial Matern GMRF via SPDE (with assigned priors)
spde = inla.spde2.pcmatern(mesh = mesh, 
                           prior.range = c(10000, 0.01), # P(range < 10 km) = 0.01
                           prior.sigma = c(1, 0.01)) #P(sd > 1) = 0.05

# Locator index for spatial random effect
spat.idx <- inla.spde.make.index('spat.idx', spde$n.spde)

# Projector matrix for spatial locations (and indexed on Year for trends)
A.spat <- inla.spde.make.A(mesh, 
                           loc=cbind(dat.train$East, dat.train$North), 
                           weights = dat.train$Year.mod)

# Make data stack
dat.stack <- inla.stack(
  data = list(y = dat.train$accum), 
  A = list(1, 1, 1, A.spat),
  effects = list(list(Intercept = rep(1, nrow(dat.train))), 
                 tibble(elev = dat.train$elev, dem=dat.train$dem, 
                        slope=dat.train$slope, aspect=dat.train$aspect),
                 list(time = dat.train$Year.mod),
                 spat.idx),
  tag = 'dat')

# Prior for temporal autocorrelation
time.spec = list(rho = list(prior = 'pc.cor1', param = c(0.3, 0.95)))

# Model formula
f.mod = y ~ -1 + Intercept + #Global intercept
  dem + #Fixed effects
  f(time, model = 'ar1', hyper = time.spec) +  #Temporal random effect modeled as AR1
  f(spat.idx, model = spde) #Spatial randomed effect (indexed by Year)

# Run model
mod = inla(f.mod,
           data = inla.stack.data(dat.stack),
           family = 'Gamma',
           control.predictor = list(compute = TRUE, A = inla.stack.A(dat.stack), link=1),
           control.compute = list(waic=TRUE, config=TRUE))

# Loop model rerun to address issues in Hessian
n.iter = 1
while (mod$mode$mode.status>0 && n.iter<=3) {
  print("Issues with Hessian. Re-running to solve negative eigenalues in the Hessian")
  n.iter = n.iter + 1
  print(paste("Starting Iteration", n.iter, "of INLA..."))
  mod = inla.rerun(mod)
}

if (mod$mode$mode.status > 0) {
  print("Issues with Hessian eigenvalues persist. Treat model results with suspicion!")
}

## MODEL VALIDATION

# Projector matrix for spatial locations (and indexed on Year for trends)
A.valid <- inla.spde.make.A(mesh, 
                            loc=cbind(dat.test$East, dat.test$North), 
                            weights = dat.test$Year.mod)

# Make data stack
valid.stack <- inla.stack(
  data = list(y = dat.test$accum), 
  A = list(1, 1, 1, A.valid),
  effects = list(list(Intercept = rep(1, nrow(dat.test))), 
                 tibble(elev = dat.test$elev, dem=dat.test$dem, 
                        slope=dat.test$slope, aspect=dat.test$aspect),
                 list(time = dat.test$Year.mod),
                 spat.idx),
  tag = 'valid')
stack.full = inla.stack(dat.stack, valid.stack)

# Generate estimates at test locations using fitted model.
mod.valid = inla(f.mod,
                 data = inla.stack.data(stack.full),
                 family = 'Gamma',
                 control.predictor = list(compute = TRUE, A = inla.stack.A(stack.full), link=1),
                 control.compute = list(return.marginals.predictor=TRUE), 
                 # control.compute = list(waic=TRUE, config=TRUE),
                 control.mode = list(theta = mod$mode$theta, restart=FALSE))

# extract indices of validation data
idx.valid <- inla.stack.index(stack.full, tag='valid')$data

# Create df of validation results
valid.df = dat.test %>% select(trace_ID, East, North, Year, accum, std, SMB) %>% 
  mutate(mu.pred=mod.valid$summary.fitted.values$mean[idx.valid], 
         mu.sd=mod.valid$summary.fitted.values$sd[idx.valid]) %>% 
  mutate(y.sd=sqrt(mu.pred^2/mod.valid$summary.hyperpar$mean[1]), 
         log.diff=log(mu.pred/accum))

# Extract indices of original model observations
idx.obs = inla.stack.index(stack.full, tag='dat')$data

# Create df of model results
results.df = dat.train %>% select(trace_ID, East, North, Year, accum, std, SMB) %>% 
  mutate(mu.pred=mod.valid$summary.fitted.values$mean[idx.obs], 
         mu.sd=mod.valid$summary.fitted.values$sd[idx.obs]) %>% 
  mutate(y.sd=sqrt(mu.pred^2/mod.valid$summary.hyperpar$mean[1]), 
         log.diff=log(mu.pred/accum))

## Predict at grid locations

# Import topo data from REMA tiles
REMA.dir = file.path("/media/durbank/WARP/Research/Antarctica/Data/DEMs/REMA/REMA_200m")
dem = rast(file.path(REMA.dir, "REMA_200m_dem.tif"))
crs(dem) = st_crs(gdf_traces)$wkt
slp = rast(file.path(REMA.dir, "REMA_200m_slope.tif"))
crs(slp) = crs(dem)
aspct = rast(file.path(REMA.dir, "REMA_200m_aspect.tif"))
crs(aspct) = crs(dem)
topo.stk = c(dem, slp, aspct)

# Prediction raster attributes
pred.bbox = st_bbox(gdf_traces)
stepsize = 1000
buffer = 5*stepsize
E.range = round(c(pred.bbox[1]-buffer, pred.bbox[3]+buffer))
N.range = round(c(pred.bbox[2]-buffer, pred.bbox[4]+buffer))
nxy <- round(c(diff(E.range), diff(N.range)) / stepsize) # Calculate the number of cells in the x and y ranges

# Build raster for gridded data
grid.blank = rast(xmin=E.range[1], xmax=E.range[2], resolution=stepsize, 
                 ymin=N.range[1], ymax=N.range[2], crs=st_crs(gdf_traces)$wkt)

# Crop topo data to grid extent
topo.stk = topo.stk %>% crop(grid.blank, snap='out')

# Resample topo data to new raster
grid.rast = resample(topo.stk, grid.blank)

# Rename rast dem layer
names(grid.rast)[1] = "DEM"

# Project the model mesh nodes onto grid
proj.grid = inla.mesh.projector(mesh,
                                 xlim = E.range,
                                 ylim = N.range,
                                 dims = nxy)

# Project trend spatial field to grid
trend.proj = inla.mesh.project(proj.grid, mod.valid$summary.random$spat.idx$mean)
trend.sd = inla.mesh.project(proj.grid, mod.valid$summary.random$spat.idx$sd)

# Add trend data to raster grid
r.tmp = rast(resolution=res(grid.rast), extent=ext(grid.rast), 
             names="trend", vals=c(trend.proj), crs=crs(grid.rast)) %>% 
  flip(direction="vertical")
add(grid.rast) <- r.tmp
r.tmp = rast(resolution=res(grid.rast), extent=ext(grid.rast), 
             names="trend.sd", vals=c(trend.sd), crs=crs(grid.rast)) %>% 
  flip(direction="vertical")
add(grid.rast) <- r.tmp

# Construct centered and scaled fixed effects
dem.fx = (as.matrix(grid.rast$DEM, wide=TRUE) - mean(data$dem))/sd(data$dem)

# Construct time random effect array
Years = 1975:2014
tmp = replicate(dim(grid.rast)[2], 
                matrix(rep(mod.valid$summary.random$time$mean,
                           each=dim(grid.rast)[1]),
                       nrow=length(Years), ncol=dim(grid.rast)[1], byrow=TRUE))
time.RE = aperm(tmp, c(2,3,1))

# Construct linear predictor for each spacetime grid cell
yr.mod = Years - mean(Years)
lin.est = mod.valid$summary.fixed$mean[1] + 
  mod.valid$summary.fixed$mean[2] * replicate(length(Years), dem.fx) + 
  sweep(replicate(length(Years), as.matrix(grid.rast$trend, wide=TRUE)), 
        MARGIN=3, yr.mod, `*`) + 
  time.RE

# Transform eta to accum estimates
pred.mu = exp(lin.est)

# Convert predictions to raster
mu.rast = rast(resolution=res(grid.rast), extent=ext(grid.rast), crs=crs(grid.rast))
for (i in 1:dim(lin.est)[3]) {
  add(mu.rast) <- rast(names=as.character(Years[i]), vals=pred.mu[,,i], 
                       resolution=res(mu.rast), extent=ext(mu.rast), 
                       crs=crs(mu.rast))
}

# Get gamma-errors for predictions
pred.sd = sqrt(pred.mu^2/mod.valid$summary.hyperpar$mean[1])

# Convert prediction errors to raster
sd.rast = rast(resolution=res(grid.rast), extent=ext(grid.rast), crs=crs(grid.rast))
for (i in 1:dim(lin.est)[3]) {
  add(sd.rast) <- rast(names=as.character(Years[i]), vals=pred.sd[,,i], 
                       resolution=res(sd.rast), extent=ext(sd.rast), 
                       crs=crs(sd.rast))
}










# Save data for later import and plotting in article
saveRDS(mod.valid, '/media/durbank/WARP/Research/Antarctica/WAIS-central/data/interim-models/mod-full.skip10.cut5.rds')
saveRDS(valid.df, '/media/durbank/WARP/Research/Antarctica/WAIS-central/data/dfs/valid.df.rds')
saveRDS(results.df, '/media/durbank/WARP/Research/Antarctica/WAIS-central/data/dfs/results.df.rds')
writeRaster(grid.rast, '/media/durbank/WARP/Research/Antarctica/WAIS-central/data/rasters/grid.tif', overwrite=TRUE)
writeRaster(mu.rast, '/media/durbank/WARP/Research/Antarctica/WAIS-central/data/rasters/mu.tif', overwrite=TRUE)
writeRaster(sd.rast, '/media/durbank/WARP/Research/Antarctica/WAIS-central/data/rasters/sd.tif', overwrite=TRUE)
