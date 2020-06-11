#%% [markdown]

# # Repeatability tests
# 
# This notebook serves as a test of the repeatability of results between different flightlines.
# To do this, we use the flightlines from 2011-11-09 in conjunction with a largely repeat flightline over the same area from 2016-11-09.

#%%
# Import requisite modules
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import time

# Set project root directory
ROOT_DIR = Path(__file__).parent.parent

# Set project data directory
DATA_DIR = ROOT_DIR.joinpath('data')