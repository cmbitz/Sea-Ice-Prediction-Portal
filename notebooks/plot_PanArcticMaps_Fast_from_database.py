
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
Plot forecast maps with all available models.
'''




import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import OrderedDict
import itertools
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
import json
from esio import EsioData as ed
from esio import ice_plot
from esio import import_data
import subprocess
import dask
from dask.distributed import Client
import timeit

# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=.8, rc={"lines.linewidth": 2.5})


# In[2]:


# # Set up local cluster for testing
# client = Client()
# client


# In[3]:


# Plotting Info
runType = 'forecast'
variables = ['sic']
metrics_all = {'sic':['anomaly','mean','SIP'], 'hi':['mean']}

Ndays = 7 # time period to aggregate maps to (default is 7, all hell could break loose if changed)

updateAll = False  # when this is false we do not remake figures done already except the last NweeksUpdate

Npers =  4  # number of weeks to potentially plot counting back from current date 
            # BUT only remake those already made if updateAll = TRUE
            # about once every few months should do an updateALL = TRUE to fill in obs and other delayed data

NweeksUpdate = 2 #3 # Always update the most recent NweeksUpdate weeks (usually 2 or 3 )



# Exclude some models
MME_NO = ['hcmr']

# Define Init Periods here, spaced by 7 days (aprox a week)
# Now
cd = datetime.datetime.now()
cd = datetime.datetime(cd.year, cd.month, cd.day) # Set hour min sec to 0. 
#cd = datetime.datetime(cd.year, 3, 4)  # force redo of period ending 2019-03-03
#cd = datetime.datetime(cd.year, 4, 20)  # force redo of 2019-04-14
#cd = datetime.datetime(cd.year, 4, 10)  # force redo of 2019-04-07

#cd = datetime.datetime(cd.year, 5, 8)  # force redo of 2019-04-07

# Hardcoded start date (makes incremental weeks always the same)
start_t = datetime.datetime(1950, 1, 1) # datetime.datetime(1950, 1, 1)
init_slice = np.arange(start_t, cd, datetime.timedelta(days=Ndays)).astype('datetime64[ns]')
init_slice = init_slice[-Npers:] # Select only the last Npers of periods (weeks) since current date



# Forecast times to plot
weeks = pd.to_timedelta(np.arange(0,5,1), unit='W')
morewks = pd.to_timedelta([9,13,17,22,26], unit='W')

slices = weeks.union(morewks).round('1d')
da_slices = xr.DataArray(slices, dims=('fore_time'))
print('forecast time slices to plot in days:')
print(da_slices.fore_time.values.astype('timedelta64[D]'))

# Help conversion between "week/month" period used for figure naming and the actual forecast time delta value
int_2_days_dict = dict(zip(np.arange(0,da_slices.size), da_slices.values))
days_2_int_dict = {v: k for k, v in int_2_days_dict.items()}


# In[4]:


# doint this temporarily to force remake of a few times in the past
#updateAll = True
#print(init_slice[16:])
#init_slice=init_slice[16:]


# In[5]:


#init_slice=init_slice[0:16]
#updateAll = True
# had Npers = 30


# In[6]:


E = ed.EsioData.load()

# add missing info for climatology, but do not use anymore
E.model_color['climatology'] = (0,0,0)
E.model_linestyle['climatology'] = '--'
E.model_marker['climatology'] = '*'
E.model['climatology'] = {'model_label':'Extrap.\n Clim. Trend'}
E.icePredicted['climatology'] = True

mod_dir = E.model_dir

# Define models to plot
models_2_plot = list(E.model.keys())
models_2_plot = [x for x in models_2_plot if x not in ['piomas','MME','MME_NEW','uclsipn']] # remove some models
models_2_plot = [x for x in models_2_plot if x not in ['modcansipns_3', 'modcansipns_4', 'szapirosipn', 'awispin', 'nicosipn']] # might return these in summer
models_2_plot = [x for x in models_2_plot if x not in ['noaasipn','noaasipn_ext']] # might return these in summer
models_2_plot = [x for x in models_2_plot if E.icePredicted[x]] # Only predictive models
models_2_plot = [x for x in models_2_plot if x not in ['usnavygofs']] # remove some models

print(models_2_plot)


# In[7]:


models_2_plot = ['MME']+models_2_plot # Add models to always plot at top
models_2_plot.insert(1, models_2_plot.pop(-1)) # Move climatology from last to second

# arrange in order of models that have longest lead times, OLD
##models_2_plot.insert(13, models_2_plot.pop(4)) # Move yopp
##models_2_plot.insert(6, models_2_plot.pop(11)) # Move NESM-ext
##models_2_plot.insert(7, models_2_plot.pop(10)) # Move KMA
##models_2_plot.insert(7, models_2_plot.pop(15)) # Move fgoalssipn

# arrange in order of models that have longest lead times
models_2_plot.insert(14, models_2_plot.pop(4)) # Move yopp
models_2_plot.insert(6, models_2_plot.pop(12)) # Move NESM-ext
models_2_plot.insert(7, models_2_plot.pop(6)) # Move MF
models_2_plot.insert(8, models_2_plot.pop(11)) # Move KMA
models_2_plot.insert(8, models_2_plot.pop(16)) # Move fgoalssipn

# switch to climo10 yrs and
# add missing info for climo10yrs for future
models_2_plot[1]='climo10yrs'
E.model_color['climo10yrs'] = (0,0,0)
E.model_linestyle['climo10yrs'] = '--'
E.model_marker['climo10yrs'] = '*'
E.model['climo10yrs'] = {'model_label':'Climatology\nLast 10 Yrs'}
E.icePredicted['climo10yrs'] = True

print(models_2_plot)


# In[8]:


models_2_plot_master ={0: models_2_plot,
                       1: models_2_plot,
                       2: models_2_plot,
                       3: models_2_plot[0:15],
                       4: models_2_plot[0:14],
                       5: models_2_plot[0:9],
                       6: models_2_plot[0:9],
                       7: models_2_plot[0:8],
                       8: models_2_plot[0:8],
                       9: models_2_plot[0:8]  }

for iweek in np.arange(0,10,1):
    print('for week  ',iweek,' models are:' ,models_2_plot_master[iweek])


# In[9]:


# Get median ice edge by DOY
median_ice_fill = xr.open_mfdataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'ice_edge.nc')).sic
# Get mean sic by DOY
mean_1980_2010_sic = xr.open_dataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'mean_1980_2010_sic.nc')).sic
# Get average sip by DOY
mean_1980_2010_SIP = xr.open_dataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'hist_SIP_1980_2010.nc')).sip    


# In[10]:


def get_figure_init_times(fig_dir):
    # Get list of all figures
    fig_files = glob.glob(os.path.join(fig_dir,'*/*.png'))
    init_times = list(reversed(sorted(list(set([os.path.basename(x).split('_')[3] for x in fig_files])))))
    return init_times


# In[11]:


def update_status(ds_status=None, fig_dir=None, int_2_days_dict=None, NweeksUpdate=3):
    # function to populate the status with 1 if the fig has been made or leave it set to nan if unmade
    # Get list of all figures
    fig_files = glob.glob(os.path.join(fig_dir,'*.png'))
    # For each figure
    for fig_f in fig_files:
        # Get the init_time from file name
        cit = os.path.basename(fig_f).split('_')[3]
        # Get the forecast int from file name
        cft = int(os.path.basename(fig_f).split('_')[4].split('.')[0])
        # Check if current it and ft were requested, otherwise skip
        if (np.datetime64(cit) in ds_status.init_time.values) & (np.timedelta64(int_2_days_dict[cft]) in ds_status.fore_time.values):
            # Always update the last 3 weeks (some models have lagg before we get them)
            # Check if cit is one of the last NweeksUpdate init times in init_time
            if (np.datetime64(cit) not in ds_status.init_time.values[-NweeksUpdate:]):
                ds_status.status.loc[dict(init_time=cit, fore_time=int_2_days_dict[cft])] = 1
        
    return ds_status

# this bit is just to try to figure out what this code does
DontSkipThis = False

if DontSkipThis:

    cvar = 'sic'
    fig_dir = os.path.join(E.fig_dir, 'model', 'all_model', cvar, "Regional_maps_NEW")


    ds_status = xr.DataArray(np.ones((init_slice.size, da_slices.size))*np.NaN, 
                                 dims=('init_time','fore_time'), 
                                 coords={'init_time':init_slice,'fore_time':da_slices}) 
    ds_status.name = 'status'
    ds_status = ds_status.to_dataset()

    print(ds_status)

    print('ds_status.init_time.values ',ds_status.init_time.values)
    print('ds_status.fore_time.values ',ds_status.fore_time.values)
    print('ds_status.fore_time.values ',ds_status.status.values)

    # Check what plots we already have
    print("Set status to 1 for figures we have already made")

    # Get list of all figures
    fig_files = glob.glob(os.path.join(fig_dir,'*.png'))
#    print(fig_files)
    # For each figure
    for fig_f in fig_files:
        # Get the init_time from file name
        cit = os.path.basename(fig_f).split('_')[4]
        # Get the forecast int from file name
        #print(fig_f)
        cft = int(os.path.basename(fig_f).split('_')[5].split('.')[0])
        #print(cit, cft)
        # Check if current it and ft were requested, otherwise skip
        if (np.datetime64(cit) in ds_status.init_time.values) & (np.timedelta64(int_2_days_dict[cft]) in ds_status.fore_time.values):
            # Always update the last 3 weeks (some models have lagg before we get them)
            # Check if cit is one of the last NweeksUpdate init times in init_time
            if (np.datetime64(cit) not in ds_status.init_time.values[-NweeksUpdate:]):
                ds_status.status.loc[dict(init_time=cit, fore_time=int_2_days_dict[cft])] = 1


    print(ds_status.status.values)

    # Drop IC/FT for figures we have already made
    ds_status = ds_status.where(ds_status.status.sum(dim='fore_time')<ds_status.fore_time.size, drop=True)

    print('ds_status.init_time.values ',ds_status.init_time.values)
    print('ds_status.fore_time.values ',ds_status.status.values)




# In[12]:


ds_region = xr.open_dataset(os.path.join(E.grid_dir, 'sio_2016_mask_Update.nc'))
print(ds_region)
print(ds_region.region_names)

reg2plot = (2,3,4,6,7,8,9,10,11,12,13,15)
# Okhotsk, Bering, Hudson, Baffin+E. Grn., Barents+Kara, Laptev+E. Sib., Chuk+Beauf, Central
reg2plot = (2,3,4,(6,7),(8,9),(10,11),(12,13),15)

print(reg2plot)


# In[13]:


init_slice


# In[17]:


def Update_PanArctic_Maps():
    
    # Make requested dataArray as specified above
    ds_status = xr.DataArray(np.ones((init_slice.size, da_slices.size))*np.NaN, 
                             dims=('init_time','fore_time'), 
                             coords={'init_time':init_slice,'fore_time':da_slices}) 
    ds_status.name = 'status'
    ds_status = ds_status.to_dataset()


    # Check what plots we already have
    if not updateAll:
        print("set status to 1 for figures we have already made, so we do not remake. Just make nans")
        ds_status = update_status(ds_status=ds_status, fig_dir=fig_dir, 
                                  int_2_days_dict=int_2_days_dict, 
                                  NweeksUpdate=NweeksUpdate)

        print(ds_status.status.values)
        # Drop IC/FT we have already plotted (orthoginal only)
        ds_status = ds_status.where(ds_status.status.sum(dim='fore_time')<ds_status.fore_time.size, drop=True)

    print("Starting plots...")
    # For each init_time we haven't plotted yet
        
    for it in ds_status.init_time.values: 
        start_time_cmod = timeit.default_timer()
        print(it)
        it_start = it-np.timedelta64(Ndays,'D') + np.timedelta64(1,'D') # Start period for init period (it is end of period). Add 1 day because when
        # we select using slice(start,stop) it is inclusive of end points. So here we are defining the start of the init AND the start of the valid time.
        # So we need to add one day, so we don't double count. 

        # For each forecast time we haven't plotted yet
        ft_to_plot = ds_status.sel(init_time=it)
        ft_to_plot = ft_to_plot.where(ft_to_plot.isnull(), drop=True).fore_time
        
        print('all forecast times to be plotted, ft_to_plot ',ft_to_plot.values)

        for ft in ft_to_plot.values: 

            print('Processing forecast time: ',ft.astype('timedelta64[D]'))

            ift = days_2_int_dict[ft] # index of ft
            cs_str = format(days_2_int_dict[ft], '02') # Get index of current forcast week
            week_str = format(iweek , '02') # Get string of current week
            cdoy_end = pd.to_datetime(it + ft).timetuple().tm_yday # Get current day of year end for valid time
            cdoy_start = pd.to_datetime(it_start + ft).timetuple().tm_yday  # Get current day of year end for valid time
            it_yr = str(pd.to_datetime(it).year) 
            it_m = str(pd.to_datetime(it).month)

            # Get datetime64 of valid time start and end
            valid_start = it_start + ft
            valid_end = it + ft
            
            print(ift)
            #if ift<=0:
            #    continue

            models_2_plot=models_2_plot_master[ift]
            print('models to plot ',models_2_plot)
            
            Nmod = len(models_2_plot) + 2  
            if ift==0:
                Nc = int(np.floor(np.sqrt(Nmod)))
                # Max number of columns == 5 (plots get too small otherwise)
                Nc = 5 #np.min([Nc,5])
                Nr = int(np.ceil((Nmod-1)/Nc))
                print(Nr, Nc, Nmod)
                assert Nc*Nr>=Nmod-1, 'Need more subplots'

            # Loop through variable of interest + any metrics (i.e. SIP) based on that
            for metric in metrics_all[cvar]:

                # Set up plotting info
                if cvar=='sic':
                    if metric=='mean':
                        cmap_c = matplotlib.colors.ListedColormap(sns.color_palette("Blues_r", 10))
                        cmap_c.set_bad(color = 'lightgrey')
                        c_label = 'Sea Ice Concentration (-)'
                        c_vmin = 0
                        c_vmax = 1
                    elif metric=='SIP':
                        cmap_c = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","orange","red","#990000"])
                        cmap_c.set_bad(color = 'lightgrey')
                        c_label = 'Sea Ice Probability (-)'
                        c_vmin = 0
                        c_vmax = 1
                    elif metric=='anomaly':
#                         cmap_c = matplotlib.colors.ListedColormap(sns.color_palette("coolwarm", 9))
                        cmap_c = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","white","blue"])
                        cmap_c.set_bad(color = 'lightgrey')
                        c_label = 'SIC Anomaly to 1980-2010 Mean'
                        c_vmin = -1
                        c_vmax = 1

                elif cvar=='hi':
                    if metric=='mean':
                        cmap_c = matplotlib.colors.ListedColormap(sns.color_palette("Reds_r", 10))
                        cmap_c.set_bad(color = 'lightgrey')
                        c_label = 'Sea Ice Thickness (m)'
                        c_vmin = 0
                        c_vmax = None
                else:
                    raise ValueError("cvar not found.") 


                # New Plot
                (f, axes) = ice_plot.multi_polar_axis(ncols=Nc, nrows=Nr, 
                                              Nplots=Nmod)


                ############################################################################
                #                               OBSERVATIONS                               #
                ############################################################################

                # Plot Obs (if available)
                ax_num = 0
                axes[ax_num].set_title('Observed', fontsize=10)

                try:
                    da_obs_c = ds_ALL[metric].sel(model=b'Observed',init_end=it, fore_time=ft)
                    #print('da_obs_c',da_obs_c)
                    haveObs = True # we think there are obs...
                except KeyError:
                    haveObs = False

                rightnow = datetime.datetime.now()
                if valid_start > np.datetime64(rightnow):
                    haveObs = False  # but we know there are no obs in the future...

                # another brute force method
                nonnancount = np.count_nonzero(~np.isnan(da_obs_c.values))
                if nonnancount == 0:
                    haveObs = False  # no obs

                # If obs then plot
                if haveObs:
                    #da_obs_c = da_obs_c.where(ds_region.mask.isin(reg2plot[cR]))
                    da_obs_c.plot.pcolormesh(ax=axes[ax_num], x='lon', y='lat', 
                                          transform=ccrs.PlateCarree(),
                                          add_colorbar=False,
                                          cmap=cmap_c,
                                          vmin=c_vmin, vmax=c_vmax)
                    axes[ax_num].set_title('Observed', fontsize=10)     
                else: # When in the future (or obs are missing)
                    #print('no obs avail yet')
                    if metric=='SIP': # Plot this historical mean SIP 
                        print("plotting hist obs SIP")
                        da_obs_c = mean_1980_2010_SIP.isel(time=slice(cdoy_start,cdoy_end)).mean(dim='time')
                        da_obs_c.plot.pcolormesh(ax=axes[ax_num], x='lon', y='lat', 
                          transform=ccrs.PlateCarree(),
                          add_colorbar=False,
                          cmap=cmap_c,
                          vmin=c_vmin, vmax=c_vmax)
                        axes[ax_num].set_title('Climatology\nLast 1980-2010', fontsize=10)
                    else:
                        textstr = 'Not Available'
                        # these are matplotlib.patch.Patch properties
                        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

                        # place a text box in upper left in axes coords
                        axes[ax_num].text(0.075, 0.55, textstr, transform=axes[ax_num].transAxes, fontsize=8,
                                verticalalignment='top', bbox=props)

                ############################################################################
                #                    Plot all models                                       #
                ############################################################################
                p = {}
                i=0
                for (i_nouse, cmod) in enumerate(models_2_plot):
                    #print(cmod)
                    i = i+1 # shift for obs
                    axes[i].set_title(E.model[cmod]['model_label'], fontsize=10)

                    # Select current model to plot
                    try:
                        ds_model = ds_ALL[metric].sel(model=cmod.encode('utf-8'),init_end=it, fore_time=ft)
                        haveMod = True
                    except:
                        haveMod = False

                    # another brute force method
                    nonnancount = np.count_nonzero(~np.isnan(ds_model.values))
                    if nonnancount == 0:
                        haveMod = False  # no output

                    # Plot
                    if haveMod:
                        # Select region
                        # Lat and Long feilds have round off differences, so set to same here
                        ds_model['lat'] = ds_region.lat
                        ds_model['lon'] = ds_region.lon
                        ds_model = ds_model.where(ds_region.mask<20, other = np.nan)

                        p[i] = ds_model.plot.pcolormesh(ax=axes[i], x='lon', y='lat', 
                                          transform=ccrs.PlateCarree(),
                                          add_colorbar=False,
                                          cmap=cmap_c,
                                          vmin=c_vmin, vmax=c_vmax)

                        axes[i].set_title(E.model[cmod]['model_label'], fontsize=10)

                        # Clean up for current model
                        ds_model = None

                    else:
                        textstr = 'Not Available'
                        # these are matplotlib.patch.Patch properties
                        props = dict(boxstyle='round', facecolor='white', alpha=0.5)

                        # place a text box in upper left in axes coords
                        axes[i].text(0.075, 0.55, textstr, transform=axes[i].transAxes, fontsize=8,
                                verticalalignment='top', bbox=props)


                # Make pretty
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                if p: # if its not empty
                    cbar = f.colorbar(next(iter(p.values())), cax=cbar_ax, label=c_label) # use first plot to gen colorbar
                    if metric=='anomaly':
                        cbar.set_ticks(np.arange(-1,1.1,0.2))
                    else:
                        cbar.set_ticks(np.arange(0,1.1,0.1))

                # Set title of all plots
                init_time_2 =  pd.to_datetime(it).strftime('%Y-%m-%d')
                init_time_1 =  pd.to_datetime(it_start).strftime('%Y-%m-%d')
                valid_time_2 = pd.to_datetime(it+ft).strftime('%Y-%m-%d')
                valid_time_1 = pd.to_datetime(it_start+ft).strftime('%Y-%m-%d')

                titlesize=15
                #if ift<3: 
                #    titlesize=15
                #elif ift<5:
                #    titlesize=13
                #else:
                #    titlesize=11
                plt.suptitle('Initialization Time: '+init_time_1+' to '+init_time_2+'\n Valid Time: '+valid_time_1+' to '+valid_time_2,
                             fontsize=titlesize) # +'\n Week '+week_str

                plt.subplots_adjust(top=0.85)
                #if (ift>4):
                #    plt.subplots_adjust(top=0.75)
                #else:
                #    plt.subplots_adjust(top=0.85)

                # Save to file
                f_out = os.path.join(fig_dir,init_time_2[0:4],'panArctic_'+metric+'_'+runType+'_'+init_time_2+'_'+cs_str+'.png')
                f.savefig(f_out,bbox_inches='tight', dpi=200)
                print("saved ", f_out)
                #print("Figure took  ", (timeit.default_timer() - start_time_plot)/60, " minutes.")

                # Mem clean up
                p = None
                plt.close(f)
                da_obs_c = None

                #diehere

        # end of plot

        # Done with current it
        print("Took ", (timeit.default_timer() - start_time_cmod)/60, " minutes.")


    # Update json file
    json_format = get_figure_init_times(fig_dir)
    json_dict = [{"date":cd,"label":cd} for cd in json_format]

    json_f = os.path.join(fig_dir, 'plotdates_current.json')
    with open(json_f, 'w') as outfile:
        json.dump(json_dict, outfile)

    # Make into Gifs
    # TODO: make parallel, add &
#    for cit in json_format:
#        subprocess.call(str("/home/disk/sipn/nicway/python/ESIO/scripts/makeGif.sh " + fig_dir + " " + cit), shell=True)

    print("Finished plotting panArctic Maps.")


# In[18]:


init_slice


# In[ ]:


if __name__ == '__main__':
    # Start up Client
    client = Client(n_workers=8)
    # dask.config.set(scheduler='threads')  # overwrite default with threaded scheduler
    
    #############################################################
    # Load in Data
    #############################################################

    for cvar in variables:

        # Load in dask data from Zarr
        ds_ALL = xr.open_zarr(os.path.join(E.data_dir,'model/zarr',cvar+'.zarr'))

        # Define fig dir and make if doesn't exist
        fig_dir = os.path.join(E.fig_dir, 'model', 'all_model', cvar, "maps_weekly_NEW")
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        # Call function
        Update_PanArctic_Maps()


# In[ ]:


#client.close()


# In[ ]:


#cvar='sic'
#ds_ALL = xr.open_zarr(os.path.join(E.data_dir,'model/zarr',cvar+'.zarr'))


# In[ ]:


#ds_ALL.sel(model=b'climo10yrs').isel(init_end=78).isel(fore_time=4)['anomaly'].plot()
#ds_ALL.sel(model=b'climo10yrs').isel(init_end=78)


# In[ ]:


#tmp=xr.open_mfdataset('/home/disk/sipn/nicway/data/model/MME_NEW/forecast/sipn_nc/sic/mean/2019-10-13/climo10yrs/2019-11-03_climo10yrs.nc')


# In[ ]:


#tmp['mean'].plot()


