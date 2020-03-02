
# coding: utf-8

# # FGOALS Forecast
# 
# - Loads in all daily forecasts of sea ice for a given initial date (combines lead times and ensemble members, this is the slow part)
# - Regrids to polar stereographic,
# - Saves to netcdf files for each initial date

# In[1]:


'''

This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
If you use this code for a publication or presentation, please cite the reference in the README.md on the
main page (https://github.com/NicWayand/ESIO). 

Questions or comments should be addressed to nicway@uw.edu

Copyright (c) 2018 Nic Wayand

GNU General Public License v3.0


'''

# Standard Imports
#%matplotlib inline
#%load_ext autoreload
#%autoreload
import matplotlib
import scipy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import xarray as xr
import xesmf as xe
import os
from os import walk
import glob
import seaborn as sns
import pandas as pd
import datetime
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ESIO Imports
from esio import import_data

#import dask
#from dask.distributed import Client


# In[2]:


#client = Client(n_workers=8)
#client


# In[3]:


# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[4]:


# Directories
model='fgoalssipn'
runType='forecast'
base_dir = r'/home/disk/sipn/nicway/data/'
ftp_dir = r'/home/disk/sipn/upload/'
data_dir = os.path.join(base_dir, 'model', model, runType)
data_out = os.path.join(base_dir, 'model', model, runType, 'sipn_nc')
stero_grid_file = os.path.join(base_dir, 'grids', 'stereo_gridinfo.nc')


# In[5]:


# the number of ensemble members and lead times need to be the same each for each initial time
# or else our analysis will fail later
# at first this modeling group changed their file format a lot, but now they are almost always the same
# possibly the occasional variation is an upload error. Here we skip those "bad" initial times
ntimes_expecting = 65
nens_expecting = 16

# look for new data each day and concat into a manageable file

native_dir=os.path.join(data_dir,'native')
#orig_dir=os.path.join(native_dir,'orig')
orig_dir=os.path.join(ftp_dir,model,runType,'FGOALS-f2_S2S_v1.3')

print(orig_dir)

init_dates = []
for (dirpath, dirnames, filenames) in walk(orig_dir):
    init_dates.extend(dirnames)
    break
    
# remove directories that do not begin with "2"
for i in init_dates:
    if i[0] != '2':
        init_dates.remove(i)
        
print(init_dates)

for itstr in init_dates:
    itdata_dir=os.path.join(orig_dir,itstr)
    f_out = os.path.join(native_dir,model+'-'+itstr+'.nc')
    if (os.path.isfile(f_out)):
        print(f_out, ' already exists - skipping')
        continue

    ens_dirs = []
    for (dirpath, dirnames, filenames) in walk(itdata_dir):
        ens_dirs.extend(dirnames)
        break

    nens=len(ens_dirs)
    print('There are ',nens, ' ensemble members for this init time')
    
    if nens != nens_expecting:
        print('    wrong number of ensembe members, skipping this init time')
        continue
        
    print('The subdirectories with each ensemble member are: ',ens_dirs)
    
    correct_ntimes = True  # assume right number
    
    ds_list = []
    for esub in ens_dirs:
        tmp = esub.split('-')
        tmp = tmp[2]
        e=int(tmp[-2:])  
        cfiles = os.path.join(itdata_dir, esub)  
        print('Ensemble #',e,' for files:', cfiles)
        cfiles = os.path.join(cfiles, '*.cice.h1.*.nc') 
        ds = xr.open_mfdataset(cfiles, concat_dim='time')

    #    # Add ensemble coord
        ds.coords['ensemble'] = e
        da=ds.aice_d
        da=da.expand_dims('ensemble')
        ds_list.append(da)

        ntimes = len(da.time.values)
        if ntimes != ntimes_expecting:
            correct_ntimes = False  
            print('    there are not 65 lead times, skipping this init time')
            continue

    if correct_ntimes:
        print('merge to one and save file ',f_out)
        ds_all = xr.merge(ds_list)
        print(ds_all)
        ds_all.to_netcdf(f_out)


# In[6]:


obs_grid = import_data.load_grid_info(stero_grid_file, model='NSIDC')
# Ensure latitude is within bounds (-90 to 90)
# Have to do this because grid file has 90.000001
obs_grid['lat_b'] = obs_grid.lat_b.where(obs_grid.lat_b < 90, other = 90)


# In[7]:


# Regridding Options
method='nearest_s2d' # ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch']


# In[ ]:





# In[8]:


weights_flag = False # Flag to set up weights have been created
have_grid_file  = False

# CICE model variable names
varnames = ['aice_d']


# In[9]:


all_files = glob.glob(os.path.join(native_dir, '*.nc'))
#print(all_files)
print(data_out)

for cfile in all_files:
    tmp = cfile.split('-')
    itstr = tmp[1].split('.')
    itstr = itstr[0]    
    f_out=os.path.join(data_out,model+'_'+itstr+'_Stereo.nc')

    if (os.path.isfile(f_out)):
        print(f_out, ' already exists - skipping')
        continue
    
    print('itime pieces are ',itstr[0:4], itstr[4:6], itstr[6:8] )
    itime = np.datetime64(datetime.datetime(int(itstr[0:4]), int(itstr[4:6]), int(itstr[6:8])))
    print('itime is ',itime)

    ds = xr.open_mfdataset(cfile,  parallel=True)
    da = ds.aice_d
    da.coords['init_time'] = itime
    da = da.expand_dims('init_time')
    da = da.rename({'time':'fore_time'})
    #dt_mod = da.time.values[1] - da.time.values[0]
    da.coords['fore_time'] = pd.to_timedelta(np.arange(1,len(da.fore_time)+1,1), unit='D')

    da.name = 'sic'
    da.coords['lon'] = da.TLON
    da.coords['lat'] = da.TLAT
    da = da.drop(['TLAT','TLON'])
    da = da/100 # percent to fraction
    #print(da)

    # Calculate regridding matrix
    regridder = xe.Regridder(da, obs_grid, method, periodic=False, reuse_weights=weights_flag)
    weights_flag = True # Set true for following loops

    # Add NaNs to empty rows of matrix (forces any target cell with ANY source cells containing NaN to be NaN)
    if method=='conservative':
        regridder = import_data.add_matrix_NaNs(regridder)

    da_out = regridder(da)

    print('Saved file', f_out)
    # Save regridded to netcdf file
    da_out.to_netcdf(f_out)

da = None # Memory clean up


# In[10]:


# Clean up
if weights_flag:
    regridder.clean_weight_file()  # clean-up


# In[11]:


#client.close()


# # Checking and Plotting

# In[46]:


ds_new = xr.open_dataset(f_out)


# In[32]:


ds_new


# In[47]:


#plt.figure()
#ds_new.sic.sel(ensemble=11).isel(fore_time=0,init_time=0).plot()


