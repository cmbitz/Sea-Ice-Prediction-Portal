
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
Plot exetent/area from observations and models (past and future)
'''

#%matplotlib inline
#%load_ext autoreload
#%autoreload
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3
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


from esio import EsioData as ed
from esio import ice_plot
from esio import metrics

# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[2]:


# takes a long time to run, not sure why


# In[3]:


# Plotting Info
runType = 'forecast'
variables = ['sic'] #, 'hi'
metric1 = 'extent'


# In[46]:


# Initialization times to plot
cd = datetime.datetime.now()
cd = datetime.datetime(cd.year, cd.month, cd.day) # Assumes hours 00, min 00
if cd.day<22:
    if cd.month>1:
        SD = datetime.datetime(cd.year, cd.month-1, 1)  # plot init_times since 1st of previous mon
    else:
        SD = datetime.datetime(cd.year-1, 12, 1)  # plot init_times since 1st of previous mon
else:
    SD = cd - datetime.timedelta(days=35)  # plot init_times for previous 35 days 5 weeks
print(SD)
# SD = cd - datetime.timedelta(days=4*365)
# ED = cd + datetime.timedelta(days=365)


# In[47]:


# Info about models runs
# icePredicted = {'gfdlsipn':True, 'piomas':True, 'yopp':True, 'bom':False, 'cma':True, 'ecmwf':True, 
#               'hcmr':False, 'isaccnr':False, 'jma':False, 'metreofr':True, 'ukmo':True, 'eccc':False, 
#               'kma':True, 'ncep':True, 'ukmetofficesipn':True, 'ecmwfsipn':True}
# biasCorrected = 


# In[48]:


#############################################################
# Load in Data
#############################################################
E = ed.EsioData.load()


# In[49]:


# Load obs already aggregated by region
import timeit
ds_obs = xr.open_mfdataset(E.obs['NSIDC_0081']['sipn_nc']+'_yearly_agg/*.nc', concat_dim='time')
ds_obs = ds_obs.Extent
# use smoothed obs to compute damped anom
# 10 days is assumed but would be better to embed this smoothing window in alpha
# and then use it here
ds_obs_smooth = ds_obs.rolling(time=10, min_periods=1, center=True).mean()

print(ds_obs.region_names.values)


# In[50]:


# Load obs already aggregated by region, these are also computed after smoothing
# the obs with 10 day running mean
ds_climo = xr.open_mfdataset(E.obs['NSIDC_0079']['sipn_nc']+'_yearly_agg_climatology/*.nc', concat_dim='time')
ds_climo = ds_climo.ClimoTrendExtent
print(ds_climo)

# Load alphas
# 10 days is assumed but would be better to embed this smoothing window in alpha

Y_Start = 1990
Y_End = 2017
file_in = os.path.join(E.obs_dir, 'NSIDC_0079','alpha_agg', str(Y_Start)+'_'+str(Y_End)+'_Alpha.nc')
#file_in = os.path.join(E.obs_dir, c_product,'alpha_agg', 'test_Alpha3.nc')

alpha = xr.open_mfdataset(file_in, parallel=True)
alpha = alpha.alpha
print(alpha)


# In[51]:


maxleadtime=215

#print(E.model.keys())
models_2_plot = list(E.model.keys())
models_2_plot = [x for x in models_2_plot if x not in ['dampedAnomalyTrend','piomas', 'MME', 'MME_NEW']] # remove some models
print(models_2_plot)


# In[58]:


cdate = datetime.datetime.now()

# adjust colors to my liking
E.model['usnavygofs']['model_label']='NRL-GOFS'
#print(E.model_color)
#import matplotlib.colors as clr
E.model_color['usnavygofs']='tab:olive'
E.model_color['usnavyncep']='tab:olive'
E.model_color['ncep']='blue'
E.model_linestyle['usnavyncep']='-'
E.model_linestyle['ncep']='-'
E.model_linestyle['ecmwf']='-'
E.model_color['fgoalssipn']='darkblue'


# In[59]:


# list of models that have month init times, do not bunch these by week
monthly_init_model = ['gfdlsipn','ecmwfsipn', 'ukmetofficesipn','meteofrsipn', 'metreofr','awispin','nicosipn']


# In[60]:


BunchByWeek = True

def Daily2Weekly(ds_model,init_start_date):
    
    #print(ds_model.init_time)

    start_t = datetime.datetime(1950, 1, 1) # datetime.datetime(1950, 1, 1)
    init_slice = np.arange(start_t, ds_model.init_time[-1].values, datetime.timedelta(days=7)).astype('datetime64[ns]')
#    print('init_slice 1 ',init_slice)
#    print('init start date ',init_start_date)
    init_slice = init_slice[init_slice>=init_start_date] # Select only the inits after init_start_date
#    print('init_slice 2 ',init_slice)
    
    da_agg = []
    counter = 0
    for (itweek,it) in enumerate(init_slice):
        print('---- itweek',itweek, it, len(init_slice))
        if itweek<(len(init_slice)-1):
            thesedates = (ds_model.init_time>=init_slice[itweek]) & (ds_model.init_time<init_slice[itweek+1])
        else:
            thesedates = (ds_model.init_time>=init_slice[itweek]) & (ds_model.init_time<=ds_model.init_time[-1])
        thesedates = ds_model.init_time.where(thesedates, drop=True )
#        print('the dates are ', thesedates)
        if (len(thesedates)==0):
#            print('no dates in this weekly slice')
            continue
#        print('  Start date this period :',thesedates[0])
        halfwidth = (thesedates[-1]-thesedates[0])*0.5
        itweekdate = thesedates[0] + halfwidth
        itweekdate.rename({'init_time':'itweek'})
#        print('  Middle of the period date: ', itweekdate)

        da_itw =  ds_model.where(thesedates, drop=True ).mean(dim='init_time')
        da_itw.coords['itweek'] = itweek
        da_itw.coords['itweekdate'] = itweekdate

        #print(da_itw)
        # Store in list
        da_agg.append(da_itw)
        counter = counter + 1

    if counter>1:
        da_agg = xr.concat(da_agg, dim='itweek')
        da_agg=da_agg.drop('init_time')
        da_agg.coords['init_time'] = da_agg.itweekdate
        da_agg=da_agg.swap_dims({'itweek':'init_time'}) # drat the plot routine requires init_time as dim                

    else:
        da_agg = da_agg[0]
        da_agg=da_agg.drop('init_time')
        da_agg.coords['init_time'] = da_agg.itweekdate
        da_agg = da_agg.expand_dims('init_time')

    #print('\n\nda_agg\n',da_agg)

    ds_model = da_agg
    da_agg = None
    return ds_model


# # Plot Raw extents and only models that predict sea ice

# In[61]:


# cmap_c = itertools.cycle(sns.color_palette("Paired", len(E.model.keys()) ))
# linecycler = itertools.cycle(["-","--","-.",":","--"])
for cvar in variables:
    
    fig_dir = os.path.join(E.fig_dir, 'model', 'all_model', cvar, "regional_timeseries")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)

    # For each region
    for cR in ds_obs.nregions.values:
#    for cR in [6]: # ds_obs.region_names.values   

        cR_name = ds_obs.region_names.sel(nregions=cR).item(0)
        if (cR==99):
            cR_name = 'Pan-Arctic Minus Canadian Islands and St John'
            print(ds_obs.region_names.sel(nregions=5).item(0))
            print(ds_obs.region_names.sel(nregions=14).item(0))

        print(cR_name)

        # New Plot
        f = plt.figure(figsize=(15,10))
        ax1 = plt.subplot(1, 1, 1) # Observations

        for (i, cmod) in enumerate(models_2_plot):
#        for (i, cmod) in enumerate(['fgoalssipn']):

            if not E.icePredicted[cmod]:
                continue
            if ((cmod=='rasmesrl') & (cR in [99, 2, 3, 4, 5, 6, 7, 14])):  # CAFS cuts off these
                continue
            if ((cmod=='awispin') & (cR in [99, 2, 4, 5])):  # CAFS cuts off these
                continue

            print(cmod)
            # Load in Model
            model_forecast = os.path.join(E.model[cmod][runType]['sipn_nc_agg'], '*.nc')

            # Check we have files 
            files = glob.glob(model_forecast)
            if not files:
                #print("Skipping model", cmod, "no forecast files found.")
                continue # Skip this model
            ds_model = xr.open_mfdataset(model_forecast, concat_dim='init_time')
            #print('ds_model BEFORE messing with it ', ds_model)

            ds_model = ds_model.isel(fore_time=slice(0,maxleadtime)).Extent

            # Select init of interest
            init_start_date = np.datetime64(SD)   
            ds_model = ds_model.where(ds_model.init_time>= init_start_date, drop=True)
            

            #    ds_model = ds_model.where(ds_model.fore_time<np.timedelta64(maxleadtime,'D'), drop=True)
            
            if (len(ds_model.init_time)==0):  # nothing to plot
                print("Skipping model with no initial times ",cmod)
                continue

            if ((cmod not in monthly_init_model) & BunchByWeek):
                ds_model = Daily2Weekly(ds_model,init_start_date)

            #print('ds_model AFTER messing with it ', ds_model)
            # Select region
            #print('model regions ',ds_model.nregions)
            if (cR==99):
                ds_model = ds_model.sel(nregions=cR)-ds_model.sel(nregions=5)-ds_model.sel(nregions=14)
            else:
                ds_model = ds_model.sel(nregions=cR)
            
#             # Take mean of ensemble
#             ds_model = ds_model.mean(dim='ensemble')

            # Get model plotting specs
            cc = E.model_color[cmod]
            cl = E.model_linestyle[cmod]

            # Plot Model
            if i == 0: # Control only one initiailzation label in legend
                no_init_label = False
            else:
                no_init_label = True
            import timeit
            start_time = timeit.default_timer()
            
            ice_plot.plot_reforecast(ds=ds_model, axin=ax1, 
                                 labelin=E.model[cmod]['model_label'],
                                 color=cc, marker=None,
                                 linestyle=cl,
                                 no_init_label=no_init_label)
            print( (timeit.default_timer() - start_time), ' seconds.' )

            # Memeory clean up
            ds_model = None     

        cxlims = ax1.get_xlim()

       # add obs and climotrend
        if (cR==99):
            ds_obs_reg = ds_obs.sel(nregions=cR)-ds_obs.sel(nregions=5)-ds_obs.sel(nregions=14)
            ds_obs_smooth_reg = ds_obs_smooth.sel(nregions=cR)-ds_obs_smooth.sel(nregions=5)-ds_obs_smooth.sel(nregions=14)
            ds_climo_reg = ds_climo.sel(nregions=cR)-ds_climo.sel(nregions=5)-ds_climo.sel(nregions=14)
        else:
            ds_obs_reg = ds_obs.sel(nregions=cR)
            ds_obs_smooth_reg = ds_obs_smooth.sel(nregions=cR)
            ds_climo_reg = ds_climo.sel(nregions=cR)

        ds_obs_reg = ds_obs_reg.where(ds_obs.time>=np.datetime64(SD), drop=True)
        ds_obs_smooth_reg = ds_obs_smooth_reg.where(ds_obs.time>=np.datetime64(SD), drop=True)
        ds_climo_reg = ds_climo_reg.where(ds_climo.time>=np.datetime64(SD), drop=True) 
        
        # PROBABLY SHOULD SMOOTH THIS JUST AS HAVE SMOOTHED THE OBS WHEN
        # COMPUTING alpha ...
        anom = ds_obs_smooth_reg - ds_climo_reg
#        anom.plot(ax=ax1, label=str(cdate.year)+' Anom', color='y', linewidth=8)
        ds_climo_reg.plot(ax=ax1, label=str(cdate.year)+' TrendExtrap', color='gold', linewidth=6)
        ds_climo = ds_climo.where(ds_climo>0, other=0)
        # compute doy for anom dataset
        DOY = [x.timetuple().tm_yday for x in pd.to_datetime(anom.time.values)]
        cc = 'k'
        for cit in np.append(0,np.arange(29, len(DOY)-1, 30)):
            print('get alpha for init_time of ',cit)
            alp=alpha.sel(init_time=DOY[cit],nregions=cR)
            # add 1 for fore_time = 0
            alp_zero = alp[0]  # begin by stealing part of another dataarray
            alp_zero = alp_zero.expand_dims('fore_time')
            alp_zero.fore_time[0] = 0
            alp_zero.load()
            alp_zero[0] = 1.0
            alp = xr.concat([alp_zero,alp],dim='fore_time')

            damped = anom.isel(time=cit).values*alp
            damped.name = 'DampedAnom'
            tmp = anom['time'].isel(time=cit).values  # +np.timedelta64(5,'D')
            damptimes = pd.to_datetime( damped.fore_time.values, unit='D',
                                                 origin = tmp)  
            damped.coords['time'] = xr.DataArray(damptimes, dims='fore_time', coords={'fore_time':damped.fore_time})
            damped = damped.swap_dims({'fore_time':'time'})
            #print('damped ',damped)
            damped = damped + ds_climo_reg
            damped = damped.where(damped>0, other=0)
            damped.plot(ax=ax1, color=cc, linewidth=2)
            
        damped.plot(ax=ax1, label=str(cdate.year)+' DampedAnom', color=cc, linewidth=2)
        ds_obs_reg.plot(ax=ax1, label=str(cdate.year)+' Observed', color='m', linewidth=8)
        ax1.set_ylabel('Sea Ice Extent\n [Millions of square km]')

    #     # 1980-2010 Historical Interquartile Range
    #     plt.fill_between(ds_per_mean.time.values, ds_per_mean + ds_per_std, 
    #                  ds_per_mean - ds_per_std, alpha=0.35, label='1980-2010\nInterquartile Range', color='m')
        ax1.set_xlim(cxlims) # fix x limits
        cylims = ax1.get_ylim()

        # Plot current date line
        ax1.plot([cd, cd], [cylims[0], cylims[1]], color='k', linestyle='--')
        ax1.set_title(cR_name)

        # Add legend (static)
        handles, labels = ax1.get_legend_handles_labels()
        # get rid of first handle 
        labels = labels[1:]
        handles=handles[1:]
        ax1.legend(handles[::-1], labels[::-1], loc='lower right',bbox_to_anchor=(1.4, 0))

        f.autofmt_xdate()
        ax1.set_ylim(cylims)
        plt.subplots_adjust(right=.8)
        
        # Save to file
        if (cR==99):
            base_name_out = 'Region_PanArctic_'+metric1+'_'+runType+'_raw_predicted'
        else:
            base_name_out = 'Region_'+cR_name.replace(" ", "_")+'_'+metric1+'_'+runType+'_raw_predicted'
        f_out = os.path.join(fig_dir, base_name_out+'.png')
        f.savefig(f_out,bbox_inches='tight',dpi=200)
        print('saved figure ',f_out)
#        mpld3.save_html(f, os.path.join(fig_dir, base_name_out+'.html'))

        # Mem clean up
        ds_model = None
        ds_obs_reg = None
        f = None


# In[62]:


print(alpha.region_names.values)


# In[35]:





# In[54]:


labels


