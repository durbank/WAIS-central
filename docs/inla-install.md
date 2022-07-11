# INLA installation

The latest version of R (`4.1.1` as of this writing) states it is currently incompatible with R-INLA.
This seems to be some sort of bug (perhaps limited to RStudio?) as it is possible to do this.
It does require installing from a local source instead of following the [standard install instructions](https://www.r-inla.org/download-install) for R-INLA.
Binaries for the R-INLA testing version can be found at [this ftp](http://inla.r-inla-download.org/R/testing/src/contrib/), while alternative Linux builds can be found at [this related site](http://inla.r-inla-download.org/Linux-builds/).
Then simply follow [these instructions](https://riptutorial.com/r/example/5556/install-package-from-local-source) to do install from local source in RStudio.
This will also require installing some of the dependencies and misc. required R packages as well (e.g. `sp`, `foreach`, etc.).

## Update for 06 May 2022

R version is updated to `4.2.0`.
Current RStudio version is `2022.02.2`.
Currently installed INLA version: `INLA_2022.04.16`.

Despite the statement on the [official r-INLA site](https://www.r-inla.org/download-install) that no stable version exists yet for R-4.1 (and presumably R-4.2) and so the testing version is needed, it appears that the standard way to install the stable INLA version (i.e. `install.packages("INLA",repos=c(getOption("repos"),INLA="https://inla.r-inla-download.org/R/stable"), dep=TRUE)`) appears to work just fine.
This is therefore currently the recommended way to install r-INLA.
