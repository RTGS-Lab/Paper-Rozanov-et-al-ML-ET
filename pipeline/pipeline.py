import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
import rioxarray as rxr
import xarray as xr

import joblib
import lightgbm as lgb
from tqdm import tqdm

import os
from os import listdir
from os.path import join, getsize

import matplotlib.pyplot as plt

from src.PM_eq import penman_monteith
import src.config as config
from src.utils import compute_features, interpolate

import argparse
import warnings

def check_dim(ds):
    if ds.shape[1]!=1468 or ds.shape[2]!=2514:
        ds = ds[:,:1468,:2514]
    return ds

def slice_ds(ds):
    return ds.sel(
            x=slice(bbox[0], bbox[2]),
            y=slice(bbox[3], bbox[1]),  
        )

warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# 1. Get year and fnames from bash script
# 2. Check the validity of inputs (finish if something is invalid)
# 3. Produce estimation of ET and save as npz array with lat, lon and projection data

parser = argparse.ArgumentParser()
parser.add_argument('--fname', type=str)
args = parser.parse_args()

fname = args.fname
day = int(fname.split('_')[-1].split('.')[0])
year = int(fname.split('_')[-2])
root_dir = '/home/runck014/shared/et_upscaling'


bbox = [-104., 36., -82., 49.]  # MIDWEST
final_model = joblib.load(f'../models/lightgbm_model.txt')


prev_year_days = 365 if (year - 1) % 4 != 0 else 366

mcd_fname = f'{root_dir}/MCD/{year}/MCD_{year}_1.tif'
mcd = rxr.open_rasterio(mcd_fname, mask_and_scale=True).rio.reproject("EPSG:4326")
mcd = slice_ds(mcd)

arrays = []
stamps = []
for offset in range(-29, 1):
    s_day = day + offset
    if s_day > 0:
        stamps.append((year, s_day))
    else:
        stamps.append((year-1, prev_year_days + s_day))  

for fileYear, stamp in stamps:
    
    era_fname = f'{root_dir}/ERA5/{fileYear}/ERA5_{fileYear}_{stamp}.tif'
    modis_fname = f'{root_dir}/MODIS/{fileYear}/MODIS_{fileYear}_{stamp}.tif'
    modis_clouds_fname = f'{root_dir}/MODIS_Clouds/{fileYear}/MODIS_Clouds_{fileYear}_{stamp}.tif'

    era = rxr.open_rasterio(era_fname, mask_and_scale=True)
    try:
        mod = rxr.open_rasterio(modis_fname, mask_and_scale=True).rio.reproject("EPSG:4326")
        mod = slice_ds(mod)
    except Exception:
        """If the current file is incorrect or missed, use the reference one filled with nans to match projection with ERA"""
        ref_file = '/home/runck014/shared/et_upscaling/MODIS/2018/MODIS_2018_365.tif'
        mod = rxr.open_rasterio(ref_file, mask_and_scale=True).rio.reproject("EPSG:4326")
        mod = slice_ds(mod)
        mod[:] = np.nan
    try:
        mod_clouds = rxr.open_rasterio(modis_clouds_fname, mask_and_scale=True).rio.reproject("EPSG:4326")
        mod_clouds = slice_ds(mod_clouds)
    except Exception:
        """If the current file is incorrect or missed, use the reference one filled with nans to match projection with ERA"""
        ref_file = '/home/runck014/shared/et_upscaling/MODIS_Clouds/2018/MODIS_Clouds_2018_365.tif'
        mod_clouds = rxr.open_rasterio(ref_file, mask_and_scale=True).rio.reproject("EPSG:4326")
        mod_clouds = slice_ds(mod_clouds)
        mod_clouds[:] = np.nan
    
    era_res = era.rio.reproject_match(mod, resampling=rasterio.enums.Resampling.bilinear)
    era_res = slice_ds(era_res)

    era_vars = list(era_res.attrs['long_name'])
    mod_vars = list(mod.attrs['long_name']) + ['Clouds']
    var_names = era_vars + mod_vars + ['IGBP']

    era_res = check_dim(era_res)
    mod = check_dim(mod)
    mod_clouds = check_dim(mod_clouds)
    
    size = mod.values.shape[1]*mod.values.shape[2]

    if mod[6].sum()>1 and pd.isna(mod.values[6]).astype(int).sum()/size*100<10:
        mod_values = mod.values.astype(np.float32)
    else:
        mod_values = np.full(mod.shape, np.nan)
    array = np.concatenate([era_res.values.astype(np.float32), mod_values, 
                            mod_clouds.values.astype(int), mcd.values.astype(int)], axis=0)
    arrays.append(array)
arrays = np.array(arrays)
arrays_interp = interpolate(arrays[:,8:-1,:,:]) #interpolate only MOD
arrays[:,8:-1,:,:] = arrays_interp
df = compute_features(arrays, era_vars, mod_vars, var_names, stamp, year, mod.y.values,  mod.x.values)
df.head().to_csv(f'{root_dir}/test.csv',index=None)
mask_invalid = df.isna().any(axis=1)
y_pred = final_model.predict(df)
y_pred[mask_invalid] = np.nan
Lv = 2.501 - 0.00237 * df['temperature_2m']
ET0 = y_pred / (Lv * 1e6 / (24 * 60 * 60))
ET0 = ET0.values.reshape(1468, 2514).astype(np.float32)

x, y = mod.x.values, mod.y.values

np.savez_compressed(
    f'{root_dir}/ET_raw/ET_{year}_{day}.npz',
    x=x, y=y,
    ET=ET0,
)