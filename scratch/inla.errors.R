# Script to reproduce weird instability errors in inla while using Gaussian model with log link

## Import libraries
library(tidyverse)
library(INLA)

# Generate synthetic exponential data
x = seq(-25,25, length=100)
A_0 = 300
k = 0.035
set.seed(0)
e = rnorm(length(x), sd=2*k)
y = A_0*exp(k*x+e)
df1 = tibble(x=x, y=y, class="OBS")

# Define formula
formula = y ~ x

# Fit inla model to data
# prior.fixed = list(mean.intercept=250, prec=)
inla.bad = inla(formula, data = df1 %>% mutate(Intercept=1), 
                family = "Gaussian", 
                control.family = list(control.link=list(model="log")), 
                control.predictor = list(compute=TRUE, link=1))

# Fit GLM to data (for comparison)
mod.glm1 = glm(formula, data = df1 %>% mutate(Intercept=1), 
               family = gaussian(link = "log"))

# Add results to df and plot
df1 = df1 %>% 
  bind_rows(tibble(x=x, y=mod.glm1$fitted.values, class="GLM")) %>% 
  bind_rows(tibble(x=x, y=inla.bad$summary.fitted.values$mean, class="INLA"))
ggplot(df1) + 
  geom_line(data=df1 %>% filter(class == "GLM"), aes(x=x, y=y), color='red', alpha=0.33) + 
  geom_point(aes(x=x,y=y, group=class, color=class), alpha=0.33)

## Rerun with different seed

# Regenerate synthetic data
set.seed(2)
e = rnorm(length(x), sd=2*k)
y = A_0*exp(k*x+e)
df2 = tibble(x=x, y=y, class="OBS")

# Refit models
inla.good = inla(formula, data = df2 %>% mutate(Intercept=1), 
                 family = "Gaussian", 
                 control.family = list(control.link=list(model="log")), 
                 control.predictor = list(compute=TRUE, link=1))
mod.glm2 = glm(formula, data = df2 %>% mutate(Intercept=1), 
               family = gaussian(link = "log"))

# Add results to df and plot
df2 = df2 %>% 
  bind_rows(tibble(x=x, y=mod.glm2$fitted.values, class="GLM")) %>% 
  bind_rows(tibble(x=x, y=inla.good$summary.fitted.values$mean, class="INLA"))
ggplot(df2) + 
  geom_line(data=df2 %>% filter(class == "GLM"), aes(x=x, y=y), color='red', alpha=0.33) + 
  geom_point(aes(x=x,y=y, group=class, color=class), alpha=0.5)
