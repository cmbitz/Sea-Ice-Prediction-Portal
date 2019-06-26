
# coding: utf-8

# In[ ]:


'''

This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
If you use this code for a publication or presentation, please cite the reference in the README.md on the
main page (https://github.com/NicWayand/ESIO). 

Questions or comments should be addressed to nicway@uw.edu

Copyright (c) 2018 Nic Wayand

GNU General Public License v3.0


'''

# Standard Imports



import matplotlib
import scipy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import os
import glob
import seaborn as sns
import warnings
import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)

# ESIO Imports

from esio import EsioData as ed
from esio import import_data


# In[ ]:


# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[ ]:


E = ed.EsioData.load()
# Directories
all_models=['usnavygofs','usnavyncep','usnavysipn']
runType='forecast'
updateall = False


# In[ ]:


stero_grid_file = E.obs['NSIDC_0051']['grid']
obs_grid = import_data.load_grid_info(stero_grid_file, model='NSIDC')
# Ensure latitude is within bounds (-90 to 90)
# Have to do this because grid file has 90.000001
obs_grid['lat_b'] = obs_grid.lat_b.where(obs_grid.lat_b < 90, other = 90)


# In[ ]:


# Regridding Options
# method='conservative_normed' # ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch']
method = 'nearest_s2d'


# In[ ]:


# Set models that are different
var_dic = {'aice':'sic'}


# In[ ]:


for model in all_models:
    print('Regridding ', model, '...')
    
    data_dir = E.model[model][runType]['native']
    data_out = E.model[model][runType]['sipn_nc']
    model_grid_file = E.model[model]['grid']
    
    # Files are stored as per time step (about 45 per init_time)
    # First parse files to see what unique init_times we have
    # ARCu0.08_121_2018042112_t0300.nc
    prefix = 'ARCu0'
    all_files = glob.glob(os.path.join(data_dir, '*'+prefix+'*.nc'))
    if model=='usnavygofs':
        init_N = 4
    else:
        init_N = 2
    init_times = list(set([s.split('_')[init_N] for s in all_files]))
    
    print("Found ",len(init_times)," initialization times.")
    if updateall:
        print("Updating all files...")
    else:
        print("Only updating new files")


    weights_flag = False # Flag to set up weights have been created

    # Load land/sea mask file
    if os.path.basename(model_grid_file)!='MISSING':
        ds_mask = xr.open_mfdataset(model_grid_file)
    else:
        ds_mask = None

    for cf in sorted(init_times):
        # Check if already imported and skip (unless updateall flag is True)
        f_out = os.path.join(data_out, prefix+'_'+cf+'_Stereo.nc') # netcdf file out 
        if not updateall:
            # TODO: Test if the file is openable (not corrupted)
            if os.path.isfile(f_out):
                print("Skipping ", cf, " already imported.")
                continue # Skip, file already imported

        c_files = sorted(glob.glob(os.path.join(data_dir, '*'+prefix+'*_'+cf+'*.nc')))

        if len(c_files)==1:
            print("Skipping ",c_files," because this routines fails when only one file, fix later")
            continue                    
        
        # Some files have a "tau" variable that is hours since analysis
        try:
            ds = xr.open_mfdataset(c_files, concat_dim='time', decode_times=False, autoclose=True)

            # Format times
            ds.coords['init_time'] = np.datetime64(ds.tau.attrs['time_origin'])
            ds.coords['tau'] = ds.tau
            ds.swap_dims({'time':'tau'}, inplace=True)
            ds.rename({'tau':'fore_time'}, inplace=True)
            ds.fore_time.attrs['units'] = 'Forecast offset from initial time'
            ds = ds.drop(['time'])
            ds.coords['fore_time'] = ds.fore_time.astype('timedelta64[h]') 
            
        # Some do not
        except AttributeError:
            
            ds = xr.open_mfdataset(c_files, concat_dim='time', decode_times=True, autoclose=True)
            dt_mod = ds.time.values[1] - ds.time.values[0]
            ds.coords['init_time'] = ds.time.values[0] - dt_mod
            ds.coords['fore_time'] = ds.time - ds.init_time
            ds.swap_dims({'time':'fore_time'}, inplace=True);
            ds = ds.drop('time')
            
        # Rename variables per esipn guidelines
        ds.rename(var_dic, inplace=True);
        
        # Apply masks (if available)
        if ds_mask:
            print('found mask')
            # land_mask is the fraction of native grid cell that is land
            # (1-land_mask) is fraction ocean
            # Multiply sic by fraction ocean to get actual native grid cell sic
            # Also mask land out where land_mask==1
            ds = ds * (1 - ds_mask.land_mask.where(ds_mask.land_mask<1))
            
        # Add mask variable so conservative regridding works as expected
        # DOESN"T WORK WITH OTHER METHODS!!
        #ds['mask'] = ds.sic.isel(fore_time=0).notnull() # Hardcoded variable choice
                        
        # Calculate regridding matrix
        regridder = xe.Regridder(ds, obs_grid, method, periodic=False, reuse_weights=weights_flag)

        weights_flag = True # Set true for following loops

        # Add NaNs to empty rows of matrix (forces any target cell with ANY source cells containing NaN to be NaN)
        #if method=='conservative':
        #    regridder = import_data.add_matrix_NaNs(regridder)

        # Regrid variables

        var_list = []
        for cvar in ds.data_vars:
            # 0 to NaN hack
            #offset = 10
            #da_coarse = regridder(ds[cvar]+10)
            #da_coarse = da_coarse.where(da_coarse>(offset)) - offset
            #var_list.append(da_coarse)
            
            # When doing nearest neighbor
            da_coarse = regridder(ds[cvar])
            var_list.append(da_coarse)
            
        ds_out = xr.merge(var_list)

        # Expand dims
        ds_out = import_data.expand_to_sipn_dims(ds_out)
                
        # # Save regridded to netcdf file
        ds_out.to_netcdf(f_out)
        
        ds_out = None # Memory clean up
        print('Saved ', f_out)


# In[ ]:


# Clean up
if weights_flag:
    regridder.clean_weight_file()  # clean-up


# # Plotting

# In[ ]:


# sic_all = xr.open_mfdataset(f_out)

# # Set up plotting info
# cmap_sic = matplotlib.colors.ListedColormap(sns.color_palette("Blues", 10))
# cmap_sic.set_bad(color = 'red')

# # Plot original projection
# plt.figure(figsize=(20,10))
# ax1 = plt.axes(projection=ccrs.PlateCarree())
# ds_p = ds.sic.isel(fore_time=79)
# ds_p.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
#                                  vmin=0, vmax=1,
#                                  cmap=matplotlib.colors.ListedColormap(sns.color_palette("Blues", 10)),
#                     transform=ccrs.PlateCarree());
# ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
# gl = ax1.gridlines(crs=ccrs.PlateCarree(), linestyle='-')
# gl.xlabels_bottom = True
# gl.ylabels_left = True
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# ax1.coastlines(linewidth=0.75, color='black', resolution='50m');

# # Plot SIC on target projection
# (f, ax1) = ice_plot.polar_axis()
# ds_p2 = sic_all.sic.isel(init_time=0).isel(fore_time=79).isel(ensemble=0)
# ds_p2.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
#                                      transform=ccrs.PlateCarree(),
#                                      cmap=cmap_sic)
# ax1.set_title('Target Grid')


