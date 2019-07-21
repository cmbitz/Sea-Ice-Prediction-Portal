
# coding: utf-8

# # FGOALS Forecast
# 
# - Loads in all daily forecasts of sea ice
# - Regrids to polar stereographic,
# - Saves to netcdf files grouped by month of initial date

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

import dask
#from dask.distributed import Client


# In[23]:


#client = Client(n_workers=8)
#client


# In[25]:


# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[26]:


# Directories
model='fgoalssipn'
runType='forecast'
base_dir = r'/home/disk/sipn/nicway/data/'
ftp_dir = r'/home/disk/sipn/upload/'
data_dir = os.path.join(base_dir, 'model', model, runType)
data_out = os.path.join(base_dir, 'model', model, runType, 'sipn_nc')
stero_grid_file = os.path.join(base_dir, 'grids', 'stereo_gridinfo.nc')


# In[27]:


updateall = False


# In[28]:





# In[29]:


# look for new data each day and concat into a manageable file

native_dir=os.path.join(data_dir,'native')
orig_dir=os.path.join(native_dir,'orig')
print(orig_dir)

init_dates = []
for (dirpath, dirnames, filenames) in walk(orig_dir):
    init_dates.extend(dirnames)
    break
    
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
    print(nens)
    print(ens_dirs)
    tmp = ens_dirs[0].split('-')
    e=int(tmp[2])

    ds_list = []
    for esub in ens_dirs:
        tmp = esub.split('-')
        e=int(tmp[2])
        cfiles = os.path.join(itdata_dir, esub, '*.cice.h1.*.nc')  
        print(cfiles)
        ds = xr.open_mfdataset(cfiles, concat_dim='time', chunks={'time': 1, 'nj': 112, 'ni': 320} , parallel=True)

    #    # Add ensemble coord
        ds.coords['ensemble'] = e
        da=ds.aice_d
        da=da.expand_dims('ensemble')
        #print(da)
        ds_list.append(da)

    #print(ds_list)    

    print('merge to one and save file ',f_out)
    ds_all = xr.merge(ds_list)
    print(ds_all)
    ds_all.to_netcdf(f_out)


# In[32]:


obs_grid = import_data.load_grid_info(stero_grid_file, model='NSIDC')
# Ensure latitude is within bounds (-90 to 90)
# Have to do this because grid file has 90.000001
obs_grid['lat_b'] = obs_grid.lat_b.where(obs_grid.lat_b < 90, other = 90)


# In[33]:


# Regridding Options
method='nearest_s2d' # ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch']


# In[76]:





# In[ ]:


weights_flag = False # Flag to set up weights have been created
have_grid_file  = False


# In[69]:


# CICE model variable names
varnames = ['aice_d']

cd = datetime.datetime.now()
thisyear = cd.year
thismonth = cd.month
print(thisyear,thismonth)


# In[80]:


# Always import the most recent two months of files (because they get updated)    
for im in [thismonth-1,thismonth]:
    cm=im    
    if im==0:
        cm=12
        year=thisyear-1
    else:
        cm = im
        year = thisyear

    f_out = os.path.join(data_out, model+'_'+str(year)+'_'+format(cm,'02')+'_Stereo.nc')
    print('Working on ',f_out)   
    # Check any files for this year exist:
    cfiles=glob.glob(os.path.join(data_dir, 'native','*'+str(year)+format(cm, '02')+'*.nc'))
    cfiles = sorted(cfiles) 
    if (len(cfiles)==0):
        print("Skipping since no files found for year and month", year, cm, ".")
        continue    

    print("Procesing year ", year, cm)
    print(cfiles)

    da_all = []
    for ifile in cfiles:
        print(ifile)
        tmp = ifile.split('-')
        tmp = tmp[1].split('.')
        date=tmp[0]
        itime = np.datetime64(datetime.datetime(int(date[0:4]), int(date[5:6]), int(date[7:8])))
        # print(np.datetime64(date)) # this might have worked
        ds = xr.open_mfdataset(ifile,  parallel=True)
        da = ds.aice_d
        da.coords['init_time'] = itime
        da = da.expand_dims('init_time')
        da = da.rename({'time':'fore_time'})
        #dt_mod = da.time.values[1] - da.time.values[0]
        da.coords['fore_time'] = pd.to_timedelta(np.arange(1,len(da.fore_time)+1,1), unit='D')
        da_all.append(da)

    da_all = xr.concat(da_all, dim='init_time')
    da_all.name = 'sic'
    da_all.coords['lon'] = da_all.TLON
    da_all.coords['lat'] = da_all.TLAT
    da_all = da_all.drop(['TLAT','TLON'])
    da_all = da_all/100 # percent to fraction
    print(da_all)

    # Calculate regridding matrix
    regridder = xe.Regridder(da_all, obs_grid, method, periodic=False, reuse_weights=weights_flag)
    weights_flag = True # Set true for following loops

    # Add NaNs to empty rows of matrix (forces any target cell with ANY source cells containing NaN to be NaN)
    if method=='conservative':
        regridder = import_data.add_matrix_NaNs(regridder)

    da_out = regridder(da_all)
    # Save regridded to netcdf file
    da_out.to_netcdf(f_out)

    # Save regridded to multiple netcdf files by month
#         months, datasets = zip(*ds_out_all.groupby('init_time.month'))
#         paths = [os.path.join(data_out, 'GFDL_FLOR_'+str(year)+'_'+str(m)+'_Stereo.nc') for m in months]
#         xr.save_mfdataset(datasets, paths)

    da_all = None # Memory clean up
    print('Saved file', f_out)


# In[81]:


# Clean up
if weights_flag:
    regridder.clean_weight_file()  # clean-up


# In[91]:


#client.close()


# # Plotting

# In[82]:


#ds_new = xr.open_dataset(f_out)


# In[83]:


#ds_new


# In[90]:


#plt.figure()
#ds_new.sic.sel(ensemble=11).isel(fore_time=0,init_time=3).plot()

