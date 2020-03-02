
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

'''
For the bootstrap and latest nrt weekly mean obs since 1990 (it was made weekly in Agg_NSIDC_Obs)
filter with a LOESS smoother in time then polynomial fit to get the 
fit parameters. Save the fit parameters since this takes forever.

Later read in fit parameters to extrapolate forward, giving climatological trend 
benchmark

Also use fit parameters for each year to compute an anomaly and save that too for computing alpha

At present this routine is meant to be used once a year, but should make it so that it produces a new climotrend
estimate each week!!!

'''




import matplotlib
import matplotlib.pyplot as plt
from collections import OrderedDict
import numpy as np
import numpy.ma as ma
import pandas as pd
import struct
import os
import xarray as xr
import glob
import datetime
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn as sns
np.seterr(divide='ignore', invalid='ignore')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from esio import EsioData as ed
from esio import ice_plot
from esio import import_data
from esio import metrics
import dask

# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=.8, rc={"lines.linewidth": 2.5})


# # Climo10yrs is average SIC or presence for the past 10 years after first computing weekly means starting with Jan 1 each year, then interpolating linearly in time to init_time 

# In[2]:


# from dask.distributed import Client
# client = Client(n_workers=2)
# client = Client()
# client
dask.config.set(scheduler='threads')  # overwrite default with threaded scheduler
# dask.config.set(scheduler='processes')  # overwrite default with threaded scheduler


# In[3]:


# Parameters
today_date = datetime.datetime.now()
nextyear = today_date.year + 1
E = ed.EsioData.load()
mod_dir = E.model_dir
cmod = 'climo10yrs'
runType = 'forecast'


# In[4]:


#for pred_year in np.arange(1995,2020,1):   # done once to make reforecast
for pred_year in [nextyear]:
    start_year = pred_year - 10
    print(start_year,pred_year)  # normally we are computing the fits for predicting far into the future

    #############################################################
    # Load in Data that have already been averaged for each week of the year, always starting on Jan 1
    #############################################################

    # BE SURE THESE ARE NON OVERLAPPING, MUST BE UPDATED FOR NEW DATA
    # Get bootstrap and nrt observations with pole hole already filled in
    ds_81 = xr.open_mfdataset(E.obs['NSIDC_0081']['sipn_nc']+'_yearly_byweek/*byweek.nc', concat_dim='time', autoclose=True, parallel=True).sic
    #ds_51 = xr.open_mfdataset(E.obs['NSIDC_0051']['sipn_nc']+'_yearly_byweek/*byweek.nc', concat_dim='time', autoclose=True, parallel=True)
    ds_79 = xr.open_mfdataset(E.obs['NSIDC_0079']['sipn_nc']+'_yearly_byweek/*byweek.nc', concat_dim='time', autoclose=True, parallel=True).sic

    ds_79=ds_79.sel(time=slice(str(start_year),str(pred_year-1)))  # end year just has to be way in the future
    ds_81=ds_81.sel(time=slice('2015',str(pred_year-1)))  # restrict to before prediciton year, lower year not important

    # Combine bootstrap with NASA NRT
    da_sic = ds_79.combine_first(ds_81)  # takes ds_79 as priority
    ds_79 = None
    ds_81 = None

    #da_sic=ds_81  # for testing
    # add year coordinate
    year_all = [x.year for x in pd.to_datetime(da_sic.time.values)]
    da_sic.coords['year'] = xr.DataArray(year_all, dims='time', coords={'time':da_sic.time})

    # put week coordinate back since combine first rubbed them out
    DOY = [x.timetuple().tm_yday for x in pd.to_datetime(da_sic.time.values)]
    weeks= np.ceil(np.divide(DOY,7))
    weeks = weeks.astype(int)
    da_sic.coords['week'] = xr.DataArray(weeks, dims='time', coords={'time':da_sic.time})
    #print(da_sic)

    # plot so we are sure this is going right
    ocnmask = da_sic.isel(time=-30).notnull()  # take a value near the end when not likely to have missing values
    ocnmask.name = 'oceanmask'

    maxweek = da_sic.sel(time=str(pred_year-1)).week.values[-1]
    print('The last week to compute is ',maxweek)

    # Convert sea ice presence
    da_sip = (da_sic >= 0.15).astype('int') # This unfortunatly makes all NaN -> zeros...
    da_sip = da_sip.where(ocnmask,other=np.nan)  
    da_sip.coords['week'] = da_sic.week
    #print(ds_sp)
    
    ds_sip_climo = da_sip.groupby('week').mean(dim='time').to_dataset(name='SIP')
    ds_sic_climo = da_sic.groupby('week').mean(dim='time').to_dataset(name='sic')
    ds_climo=xr.merge([ds_sic_climo,ds_sip_climo])
    ds_climo = ds_climo.sel(week=slice(1,maxweek))
    ds_climo.coords['time'] = xr.DataArray([datetime.datetime(pred_year,1,1) + datetime.timedelta(days=int(7*(x-1)+3)) for x in np.arange(1,maxweek+1,1)], dims='week')
    ds_climo=ds_climo.swap_dims({'week':'time'})
    
    print(ds_climo)
    file_out = os.path.join(mod_dir, cmod, runType, 'sipn_nc_yearly_byweek',str(pred_year)+'_byweek.nc')
    ds_climo.to_netcdf(file_out)
    print("Saved",file_out)


# In[5]:


PlotTest = False
if PlotTest:
    tmpsip1=ds_sip_climo.sel(week=1) # save one time at random for plot verification
    tmpsip2=ds_sip_climo.sel(week=maxweek) # save one time at random for plot verification

    # plot one time at random to ensure it is about right Nplots has to be one more than you'd think
    (f, axes) = ice_plot.multi_polar_axis(ncols=2, nrows=1, Nplots = 3, sizefcter=3)
    tmpsip1.plot.pcolormesh(cmap='Reds',ax=axes[0], x='lon', y='lat',transform=ccrs.PlateCarree())
    tmpsip2.plot.pcolormesh(cmap='Reds',ax=axes[1], x='lon', y='lat',transform=ccrs.PlateCarree())


# In[6]:


# Hardcoded start date (makes incremental weeks always the same)
start_t = datetime.datetime(1950, 1, 1) # datetime.datetime(1950, 1, 1)
# Params for this plot
Ndays = 7 # time period to aggregate maps to (default is 7)

init_start_date = np.datetime64('2018-01-01') # first date we have computed metrics
                   
#init_start_date = np.datetime64('2019-01-01') # speeds up substantially b

cd = today_date +  datetime.timedelta(days=365)

init_slice = np.arange(start_t, cd, datetime.timedelta(days=Ndays)).astype('datetime64[ns]')
# init_slice = init_slice[-Npers:] # Select only the last Npers of periods (weeks) since current date
init_slice = init_slice[init_slice>=init_start_date] # Select only the inits after init_start_date


init_midpoint = np.arange(start_t- datetime.timedelta(days=3), cd- datetime.timedelta(days=3), datetime.timedelta(days=Ndays)).astype('datetime64[ns]')
init_midpoint = init_midpoint[init_midpoint>=init_start_date] # Select only the inits after init_start_date

print('init_slices are at end of the range')
print(init_slice[0],init_slice[-1])
print('shift to midpoint of range')
print(init_midpoint[0],init_midpoint[-1])


# In[7]:


files = os.path.join(mod_dir, cmod, runType, 'sipn_nc_yearly_byweek','*_byweek.nc')
ds = xr.open_mfdataset(files, concat_dim='time', parallel=True)
cd = today_date +  datetime.timedelta(days=375)
ds=ds.sel(time=slice(str(init_start_date),str(cd)))  # end year just has to be way in the future
print(ds)


# In[8]:


ds=ds.chunk({'time': ds.time.size, 'y': 8}) # change chunk so time is one chunk, required for interp
print(ds)
dsr=ds.interp(time=init_midpoint,method= 'linear')
dsr['time']=init_slice
dsr = dsr.rename({'time':'init_end'})
dsr = dsr.drop('week')
print(dsr)
file_out = os.path.join(mod_dir, cmod, runType, 'metrics','climoSIP_SIC_2018_to_nextyear.nc')
dsr.to_netcdf(file_out)
print("Saved",file_out)


# In[9]:


limited_slice = init_slice[init_slice<=np.datetime64(today_date)] # Select only the inits after init_start_date

# just redo since june
limited_slice = limited_slice[limited_slice>=np.datetime64('2019-06-07')] # Select only the inits after init_start_date
#limited_slice = limited_slice[limited_slice<=np.datetime64('2019-08-11')] # Select only the inits after init_start_date



# In[10]:


PlotTest = False
if PlotTest:
    tmpsip1=da.isel(time=80) # save one time at random for plot verification
    tmpsip2=dar.isel(time=80) # save one time at random for plot verification
    tmpsip2=tmpsip2-tmpsip1

    # plot one time at random to ensure it is about right Nplots has to be one more than you'd think
    (f, axes) = ice_plot.multi_polar_axis(ncols=2, nrows=1, Nplots = 3, sizefcter=3)
    tmpsip1.plot.pcolormesh(cmap='Reds',ax=axes[0], x='lon', y='lat',transform=ccrs.PlateCarree())
    tmpsip2.plot.pcolormesh(cmap='Reds',ax=axes[1], x='lon', y='lat',transform=ccrs.PlateCarree())


# In[11]:


# Get mean sic by DOY
mean_1980_2010_sic = xr.open_dataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'mean_1980_2010_sic.nc')).sic
mean_1980_2010_sic


# In[ ]:


# put the metrics in the MME_NEW folder one file per it and ft 
metrics_all = ['anomaly','mean','SIP']
cvar = 'sic'
updateAll = False

# Forecast times are weekly for a year
weeks = pd.to_timedelta(np.arange(0,52,1), unit='W')
#months = pd.to_timedelta(np.arange(2,12,1), unit='M')
#years = pd.to_timedelta(np.arange(1,2), unit='Y') - np.timedelta64(1, 'D') # need 364 not 365
#slices = weeks.union(months).union(years).round('1d')
da_slices = xr.DataArray(weeks, dims=('fore_time'))
da_slices.fore_time.values.astype('timedelta64[D]')
print(da_slices)

# For each init time period
for it in limited_slice: 
    it_start = it-np.timedelta64(Ndays,'D') + np.timedelta64(1,'D') # Start period for init period (it is end of period). Add 1 day because when
    # we select using slice(start,stop) it is inclusive of end points. So here we are defining the start of the init AND the start of the valid time.
    # So we need to add one day, so we don't double count.
    print(it_start,"to",it)

    for ft in da_slices.values: 

        cdoy_end = pd.to_datetime(it + ft).timetuple().tm_yday # Get current day of year end for valid time
        cdoy_start = pd.to_datetime(it_start + ft).timetuple().tm_yday  # Get current day of year end for valid time

        # Get datetime64 of valid time start and end
        valid_start = it_start + ft
        valid_end = it + ft

        # Loop through variable of interest + any metrics (i.e. SIP) based on that
        for metric in metrics_all:

            # File paths and stuff
            out_metric_dir = os.path.join(E.model['MME_NEW'][runType]['sipn_nc'], cvar, metric)
            if not os.path.exists(out_metric_dir):
                os.makedirs(out_metric_dir) 

            out_init_dir = os.path.join(out_metric_dir, pd.to_datetime(it).strftime('%Y-%m-%d'))
            if not os.path.exists(out_init_dir):
                os.makedirs(out_init_dir)

            out_mod_dir = os.path.join(out_init_dir, cmod)
            if not os.path.exists(out_mod_dir):
                os.makedirs(out_mod_dir)     

            out_nc_file = os.path.join(out_mod_dir, pd.to_datetime(it+ft).strftime('%Y-%m-%d')+'_'+cmod+'.nc')

            # Only update if either we are updating All or it doesn't yet exist
            # OR, its one of the last 3 init times 
            if updateAll | (os.path.isfile(out_nc_file)==False) | np.any(it in limited_slice[-2:]):
                #print("    Updating...")

                # select valid time
                ds_model =  dsr.sel(init_end=valid_end)
                
                if metric=='mean': 
                    ds_model = ds_model[cvar]

                elif metric=='SIP': 
                    ds_model = ds_model.SIP
                    
                elif metric=='anomaly': # Calc anomaly in reference to mean observed 1980-2010
                    # Get climatological mean
                    da_obs_mean = mean_1980_2010_sic.isel(time=slice(cdoy_start,cdoy_end)).mean(dim='time')
                    # Calc anom
                    ds_model =  ds_model[cvar] - da_obs_mean

                else:
                    raise ValueError('metric not implemented')

                
                # Check we have all observations for this week (7)
                if ds_model.init_end.size == 1:

                    # Add Coords info
                    ds_model.name = metric
                    ds_model.coords['model'] = cmod
                    ds_model.coords['init_start'] = it_start
                    ds_model.coords['init_end'] = it
                    ds_model.coords['valid_start'] = it_start+ft
                    ds_model.coords['valid_end'] = it+ft
                    ds_model.coords['fore_time'] = ft

                    # Write to disk
                    ds_model.to_netcdf(out_nc_file)
                    # print('Saving file ',out_nc_file)
                    # Clean up for current model
                    ds_model = None
                    
                else:
                    print('Warning NO VALID TIME where there should be', it, ft)


