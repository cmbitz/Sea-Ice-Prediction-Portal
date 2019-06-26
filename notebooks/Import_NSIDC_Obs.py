
# coding: utf-8

# In[1]:


'''

This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
If you use this code for a publication or presentation, please cite the reference in the README.md on the
main page (https://github.com/NicWayand/ESIO). 

Questions or comments should be addressed to nicway@uw.edu

Copyright (c) 2018 Nic Wayand

GNU General Public License v3.0


'''

import numpy as np
import numpy.ma as ma
import os
import xarray as xr
import glob
# import loadobservations as lo
from esio import import_data
from esio import metrics
from esio import EsioData as ed

# Dirs
E = ed.EsioData.load()
data_dir = E.obs_dir

# Flags
UpdateAll = False

# Load in regional data
# Note minor -0.000004 degree differences in latitude
ds_region = xr.open_dataset(os.path.join(E.grid_dir, 'sio_2016_mask_Update.nc'))

# Products to import
product_list = ['NSIDC_0081', 'NSIDC_0051', 'NSIDC_0079']

# Version numbers (if any)
# If multiple version number files exist, it only imports the one specified below
ver_nums = {'NSIDC_0079':'v3.1','NSIDC_0081':'nrt','NSIDC_0051':'v1.1'}

ds_lat_lon = import_data.get_stero_N_grid(grid_dir=E.grid_dir)


# In[2]:


#print(ds_lat_lon)
#print(ds_region)


# In[3]:


#import matplotlib

#ds_region.mask.plot()
#print(ds_region.region_names)


# In[4]:


# Loop through each product
for c_product in product_list:
    print('Importing ', c_product, '...')

    # Find new files that haven't been imported yet
    native_dir = os.path.join(data_dir, c_product, 'native')
    os.chdir(native_dir)
    native_files = sorted(glob.glob('*'+ver_nums[c_product]+'*.bin'))
    nc_dir = os.path.join(data_dir, c_product, 'sipn_nc')
    os.chdir(nc_dir)
    nc_files = sorted(glob.glob('*.nc'))
    if UpdateAll:
        new_files = [x.split('.b')[0] for x in native_files]
        print('Updating all ', len(native_files), ' files...')
    else:
        new_files = np.setdiff1d([x.split('.b')[0] for x in native_files], 
                                 [x.split('.n')[0] for x in nc_files]) # just get file name and compare
        print('Found ', len(new_files), ' new files to import...')

    # Loop through each file
    for nf in new_files:
        
        # Load in 
        ds_sic = import_data.load_1_NSIDC(filein=os.path.join(native_dir, nf+'.bin'), product=c_product)

        # Add lat and lon dimensions
        ds_sic.coords['lat'] = ds_lat_lon.lat
        ds_sic.coords['lon'] = ds_lat_lon.lon

        # Stereo projected units (m)
        dx = dy = 25000 
        xm = np.arange(-3850000, +3750000, +dx)
        ym = np.arange(+5850000, -5350000, -dy)
        ds_sic.coords['xm'] = xr.DataArray(xm, dims=('x'))
        ds_sic.coords['ym'] = xr.DataArray(ym, dims=('y'))    

        # get lats around the pole hole
        hole_mask = ds_sic.hole_mask  
        lats = ds_sic.lat.where(hole_mask==1)
        LMAX=lats.min().values
        
        land_mask = ds_sic.sic.notnull() # use latest
        land_mask = land_mask.where(hole_mask==0, other=1)

        # avg concentration in annular ring around pole hole
        sic_polehole=ds_sic.where(land_mask & (ds_sic.lat<LMAX) & (ds_sic.lat>LMAX-1)).sic.mean().values
#        print('stuff ',sic_polehole, LMAX)

        # Old way of computing extent and area
        # ds_sic['extent'] = metrics.calc_extent(ds_sic.sic, ds_region, fill_pole_hole=True)
        # print('extent force fill it', ds_sic['extent'])
        # ds_sic['area'] = (ds_sic.sic * ds_region.area).sum(dim='x').sum(dim='y')/(10**6) # No pole hole
        # print('no fill areas ',ds_sic.area)
        
        # fill pole hole with neighbor values
        ds_sic = ds_sic.where(hole_mask==0, other=sic_polehole)
        
        # New results of extent and area after pole hole fill
        # must not have TRUE or double counts the pole hole
        # ds_sic['extent'] = metrics.calc_extent(ds_sic.sic, ds_region, fill_pole_hole=True)
        # print('double counts it', ds_sic['extent'])

        ds_sic['extent'] = metrics.calc_extent(ds_sic.sic, ds_region, fill_pole_hole=False)
#        ds_sic.sic.plot()

#         ds_sic['extent'] = ds_sic['extent'] + (ds_sic.hole_mask.astype('int') * ds_region.area).sum(dim='x').sum(dim='y')/(10**6) # Add hole
        ds_sic['area'] = (ds_sic.sic * ds_region.area).sum(dim='x').sum(dim='y')/(10**6) # No pole hole

#        print('extent and areas ',ds_sic.extent.values, ds_sic.area.values, sic_polehole, LMAX)
        # Save to netcdf file
        ds_sic.to_netcdf(os.path.join(nc_dir, nf.split('.b')[0]+'.nc'))
        ds_sic = None

#     # Calculate extent and area (saved to separte file)
#     if len(new_files) > 0 : # There were some new files
#         print('Calculating extent and area...')
#         ds_all = xr.open_mfdataset(os.path.join(nc_dir,'*.nc'), concat_dim='time', 
#                                    autoclose=True, compat='no_conflicts',
#                                    data_vars=['sic'])    
#         print('Loaded in all files...')
#         ds_all['extent'] = ((ds_all.sic>=0.15).astype('int') * ds_region.area).sum(dim='x').sum(dim='y')/(10**6)
#         ds_all['extent'] = ds_all['extent'] + (ds_all.hole_mask.astype('int') * ds_region.area).sum(dim='x').sum(dim='y')/(10**6) # Add hole
#         ds_all['area'] = (ds_all.sic * ds_region.area).sum(dim='x').sum(dim='y')/(10**6) # No pole hole
#         ds_all = ds_all[['extent','area']]
#         # Create new dir to store agg file
#         if not os.path.exists(os.path.join(data_dir, c_product, 'agg_nc')):
#             os.makedirs(os.path.join(data_dir, c_product, 'agg_nc'))
#         ds_all.to_netcdf(os.path.join(data_dir, c_product, 'agg_nc', 'panArctic.nc'))
        
    # For each Product
    print("Finished ", c_product)
    print("")

