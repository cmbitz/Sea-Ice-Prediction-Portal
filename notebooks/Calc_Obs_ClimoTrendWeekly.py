
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

#get_ipython().magic('matplotlib inline')
#get_ipython().magic('load_ext autoreload')
#get_ipython().magic('autoreload')
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
pred_year = today_date.year + 1
print(pred_year)  # normally we are computing the fits for predicting far into the future
#pred_year = 2020 # can force to do a particular year but 2018 and 2019 are done already
start_year = 1990
E = ed.EsioData.load()
mod_dir = E.model_dir
cmod = 'climatology'
runType = 'forecast'


# In[4]:


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
print(ds_81)
print(ds_79)


# In[5]:


ds_81.time[0].values, ds_81.time[-1].values


# In[6]:


ds_79.time[0].values, ds_79.time[-1].values


# In[7]:


# Combine bootstrap with NASA NRT
da_sic = ds_79.combine_first(ds_81)  # takes ds_79 as priority

#da_sic=ds_81  # for testing
# add year coordinate
year_all = [x.year for x in pd.to_datetime(da_sic.time.values)]
da_sic.coords['year'] = xr.DataArray(year_all, dims='time', coords={'time':da_sic.time})

# put week coordinate back since combine first rubbed them out
DOY = [x.timetuple().tm_yday for x in pd.to_datetime(da_sic.time.values)]
weeks= np.ceil(np.divide(DOY,7))
weeks = weeks.astype(int)
da_sic.coords['week'] = xr.DataArray(weeks, dims='time', coords={'time':da_sic.time})
print(da_sic)


# In[8]:


ds_79 = None
ds_81 = None


# In[9]:


# plot so we are sure this is going right
ocnmask = da_sic.isel(time=-30).notnull()  # take a value near the end when not likely to have missing values
ocnmask.name = 'oceanmask'
#print(ocnmask)

PlotTest = False
if PlotTest:
    tmpsic=da_sic.isel(time=30) # save one time at random for plot verification
    #tmpsic=da_sic.mean('time')
    #print(tmpsic)

    # plot one time at random to ensure it is about right Nplots has to be one more than you'd think
    (f, axes) = ice_plot.multi_polar_axis(ncols=2, nrows=1, Nplots = 3, sizefcter=3)
    tmpsic.plot.pcolormesh(cmap='Reds',ax=axes[0], x='lon', y='lat',transform=ccrs.PlateCarree())
    ocnmask.plot.pcolormesh(cmap='Reds',ax=axes[1], x='lon', y='lat',transform=ccrs.PlateCarree())


# # Climatology forecast

# In[ ]:


TestPlot = False
if TestPlot:

    # equal to code in mertics.py put here for testing
    from scipy import stats
    import statsmodels.api as sm
    from scipy.interpolate import InterpolatedUnivariateSpline

    def _fitparams(x=None, y=None, dummy=None):                                                                                           

        # Drop indices where y are missing                                                                  
        nonans = np.isnan(y)
        x_nonans = x[~nonans]
        y_nonans = y[~nonans]

        if y_nonans.size == 0:                                                                                         
            fitparm = np.empty([3]) * np.nan
        else:
            sumy = np.sum(y_nonans)
            leny = 1.0*np.size(y_nonans)
            fitparm = np.zeros(3)
            print('sum len ',sumy,leny)
            if (sumy>0. and sumy<leny):
                lowess = sm.nonparametric.lowess(y_nonans, x_nonans, frac=.3)  # higher frac is smoother

                # unpack the lowess smoothed points to their values
                lowess_y = list(zip(*lowess))[1]
                #print(lowess_y) # a smooted version of y without extrema

                fitparm = np.polyfit(x, lowess_y, 2)
            elif (sumy==leny):
                fitparm[2] = 1.0

        return fitparm

    # explore the new method
    cweek = 20

    # Select current week of year
    da_cweek = da_sic.where(da_sic.week==cweek, drop=True).swap_dims({'time':'year'})

    ytrain=da_cweek[:,200,150].values
#    ytrain=da_cweek[:,200,200].values  # has 0.9 to 1 range
#    ytrain=da_cweek[:,200,100].values   # strange high values
#    ytrain=da_cweek[:,225,130].values  # very high values
    ytrain = ytrain*0.
    ytrain = np.ones(29)
    print(ytrain)

    cyears=np.arange(start_year,pred_year,1)
#    print('cyears ',cyears)
    origpred=metrics._lrm(cyears, ytrain, pred_year)  # old method with linear fit
    pfit =_fitparams(cyears, ytrain)  # new method local for mucking
    pfit2 = metrics._lowessfit(cyears, ytrain)  # new method in metric.py
    print(pfit)
    print(pfit2)

    fitfun = np.poly1d(pfit)
    fitfun2 = np.poly1d(pfit2)
    
    newpred = fitfun(pred_year)
    newpred2 = fitfun2(pred_year)
    
    # can I reconstruct it by hand (yes)
    tmp=cyears**2*pfit[0]+cyears*pfit[1]+pfit[2]
    tmp2=cyears**2*pfit2[0]+cyears*pfit2[1]+pfit2[2]

    #x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]

    print('linear fit in red ',origpred)
    print('new fit in blue ',newpred)
    print('   should be same ',newpred2)

    f = plt.figure()
    plt.plot(cyears,ytrain,marker='o',markersize=10,color='k')
    plt.plot(pred_year,origpred,marker='o',markersize=12,color='r')
    plt.plot(pred_year,newpred,marker='o',markersize=10,color='b')
    plt.plot(pred_year,newpred2,marker='*',markersize=10,color='g')
    plt.plot(cyears,tmp,marker='o',markersize=10,color='c')
    plt.plot(cyears,tmp2,marker='o',markersize=10,color='m')

    print('green/cyan dots are quadratic fit to lowess smoothed data')


# In[ ]:


#   this worked very poorly 
TestPlot = False
if TestPlot:

    from scipy.special import logit, expit

    def _fitparams2(x=None, y=None, dummy=None):                                                                                           
        # Drop indices where y are missing                                                                  
        nonans = np.isnan(y)
        x_nonans = x[~nonans]
        y_nonans = y[~nonans]

        if y_nonans.size == 0:                                                                                         
            fitparm = np.empty([3]) * np.nan
        else:
            ytrans = logit(y_nonans)
            ytrans = np.clip(ytrans, -500, 500)
            print('ytrans ', ytrans)
            fitparm = np.polyfit(x, ytrans, 2)
        return fitparm

TestPlot = False
if TestPlot:
    # explore the new method
    cweek = 20

    # Select current week of year
    da_cweek = da_sic.where(da_sic.week==cweek, drop=True).swap_dims({'time':'year'})

    ytrain=da_cweek[:,200,150].values
#    ytrain=da_cweek[:,200,200].values  # has 0.9 to 1 range
#    ytrain=da_cweek[:,200,100].values   # strange high values
#    ytrain=da_cweek[:,225,130].values  # very high values

    print(ytrain)

    cyears=np.arange(start_year,pred_year,1)
#    print('cyears ',cyears)
    
    pfitlogit =_fitparams2(cyears, ytrain)  # new method with logit
    fitfun3 = np.poly1d(pfitlogit)
    newpred3 = expit(fitfun3(pred_year))
    logitfit = expit(fitfun3(cyears))

#    print('logit stuff ', ylogitfit, logitpred, newpred3)
    print('logitfit ',logitfit)
    
    f, axarr = plt.subplots(2, sharex=True)
    axarr[0].plot(cyears,ytrain,marker='o',markersize=10,color='k')
    axarr[0].plot(cyears,logitfit,marker='o',markersize=8,color='m')
    axarr[0].plot(pred_year,newpred3,marker='o',markersize=8,color='m')
    axarr[0].set_title('fit using logit transpformed back to normal space')
    axarr[1].plot(cyears,logit(ytrain)-0.5,marker='o',markersize=10,color='k')
    axarr[1].plot(cyears,fitfun3(cyears),marker='o',markersize=10,color='m')
    axarr[1].set_title('fit in logit space')


# In[10]:


# This is the part that fits the weekly data, after lowess smoothing
# it is pretty slow (couple of hours to fit 52 weeks)
# we have to redo each year we want to compute the ClimoTrend because we want to use
# the most up to date data as possible
maxweeks = da_sic.sel(time=slice(str(pred_year-1),str(pred_year-1))).week.max().values

for cweek in np.arange(1,maxweeks+1,1):

    file_out = os.path.join(mod_dir, cmod, runType, 'param_weekly', 
                            str(pred_year)+'_week'+format(cweek, '02')+'_'+str(start_year)+'_'+str(pred_year - 1)+'_SICfitparams.nc')

#    print(file_out)
#    diehere
    if ((os.path.isfile(file_out)) & (cweek<maxweeks)): # force redo last week each time
        print(file_out,' has already been done')
        continue

    print("Processing week ",cweek," of ",maxweeks," for predicting year ",pred_year)

    # Select current week of year
    da_cweek = da_sic.where(da_sic.week==cweek, drop=True).swap_dims({'time':'year'})
    
    # Split by train and validate years (e.g. 1990 to 2018 for pred_year of 2019)
    da_train = da_cweek.sel(year=slice(start_year, pred_year - 1)) #.where(ocnmask) made it much slower
    
#    print('send this to fit routine ',da_train.chunk({'year': -1}))
    ds_pred = metrics.LowessQuadFit(da_train.chunk({'year': -1}), 'year') # Have to rechunk year into one big one

    # tidy up the dataset
    ds_pred.coords['week'] = cweek
    ds_pred.name = 'fitparams'
    
    # Move back to actual (valid_time) space
    ds_pred = ds_pred.expand_dims('time')
    ds_pred.coords['time'] = xr.DataArray([datetime.datetime(pred_year,1,1) + datetime.timedelta(days=int(x-1)) for x in [7*cweek]], dims='time')
        
    print('output this to file ',ds_pred)

    ds_pred.load()  # load before saving forces calculation now

    # Save to disk
    ds_pred.to_netcdf(file_out)
    print("Saved",file_out)
    


# # Clim trend extrapolations

# In[11]:


test_plots = False

if test_plots:

    # plot one just to be sure it looks good
    cweek=1  # week to read in and plot

    # read in current week of year for all years
    ds_cweek = da_sic.where(da_sic.week==cweek, drop=True).swap_dims({'time':'year'})

    # Split by train and validate years (e.g. 1990 to 2018 for pred_year of 2019)
    ds_cweek = ds_cweek.sel(year=slice(start_year, pred_year - 1)) 

    # read in the fit parameters for this range of years
    file_out = os.path.join(mod_dir, cmod, runType, 'param_weekly', 
                            str(pred_year)+'_week'+format(cweek, '02')+'_'+str(start_year)+'_'+str(pred_year - 1)+'_SICfitparams.nc')

    print(file_out)
    ds = xr.open_mfdataset(file_out, autoclose=True, parallel=True)

    recons=pred_year**2*ds.fitparams.isel(pdim=0,time=0)  +  pred_year*ds.fitparams.isel(pdim=1,time=0) +  ds.fitparams.isel(pdim=2,time=0)
    #x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]
    #print(recons)
    ocnmask=recons.notnull()
#    recons=recons.where(recons>0,other=0).where(ocnmask)
#    recons=recons.where(recons<1,other=1).where(ocnmask)
    sicmean=ds_cweek.mean('year')

    (f, axes) = ice_plot.multi_polar_axis(ncols=5, nrows=1,sizefcter=2)
    recons.plot.pcolormesh(cmap='Blues',ax=axes[0], x='lon', y='lat',transform=ccrs.PlateCarree())
    axes[0].set_title('Week 1 Fit', fontsize=20)
    sicmean.plot.pcolormesh(cmap='Blues',ax=axes[1], x='lon', y='lat',transform=ccrs.PlateCarree())
    axes[1].set_title('Past Mean', fontsize=20)
    tmp = recons-sicmean
    tmp.plot.pcolormesh(cmap='RdYlBu',ax=axes[2], x='lon', y='lat',transform=ccrs.PlateCarree())
    axes[2].set_title('Difference', fontsize=20)
    ds.fitparams.isel(pdim=0,time=0).plot.pcolormesh(cmap='RdYlBu',ax=axes[3], x='lon', y='lat',transform=ccrs.PlateCarree())
    axes[2].set_title('fit param 0', fontsize=20)


# In[14]:


# Compute and Write the climo Trend for each week of the prediction year
maxweeks = da_sic.sel(time=slice(str(pred_year-1),str(pred_year-1))).week.max().values
print(maxweeks)

#maxweeks = 52
#pred_year = 2020  # want to do 2017 to 2020 need 2017 so can get one time before 2018

for cweek in np.arange(1,maxweeks+1,1):

    if pred_year==2017:  # special case of last week of 2017 made for interpolating
        if cweek<52:
            continue
        file_in = os.path.join(mod_dir, cmod, runType, 'param_weekly', 
            str(pred_year+1)+'_week'+format(cweek, '02')+'_'+str(start_year)+'_'+str(pred_year)+'_SICfitparams.nc')
            
    else:
        # read in the fit parameters for this range of years
        file_in = os.path.join(mod_dir, cmod, runType, 'param_weekly', 
            str(pred_year)+'_week'+format(cweek, '02')+'_'+str(start_year)+'_'+str(pred_year - 1)+'_SICfitparams.nc')



    file_out = os.path.join(mod_dir, cmod, runType, 'sipn_nc_weekly', 
                                str(pred_year)+'_week'+format(cweek, '02')+'_'+str(start_year)+'_'+str(pred_year - 1)+'_SIC.nc')

    print(file_out)
    print(file_in)


    if ((os.path.isfile(file_out)) & (cweek<maxweeks)): # force redo last week each time
        print(file_out,' has already been done')
        continue


    ds = xr.open_mfdataset(file_in, autoclose=True, parallel=True)

    recons=pred_year**2*ds.fitparams.isel(pdim=0,time=0)  +  pred_year*ds.fitparams.isel(pdim=1,time=0) +  ds.fitparams.isel(pdim=2,time=0)
    #x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]
    recons.name = 'ClimoTrendSIC'
    recons = recons.drop('pdim')
    ocnmask=recons.notnull()
    recons=recons.where(recons>0,other=0).where(ocnmask)
    recons=recons.where(recons<1,other=1).where(ocnmask)
    if pred_year==2017: 
        recons['time'] = recons.time.values - np.timedelta64(365,'D') 
        print(recons.time)
    
    recons.to_netcdf(file_out)
    print("Saved",file_out)


# In[13]:





# In[ ]:


# Compute anomalies for purpose of computing alpha for damped persistence
# not important to redo with more data since 28 years should be plenty to get a good estimate for 
# no doubt this could be done more elegantly 
# did not rerun after recomputing the fit params 
# the change to fitparams is pretty trivial 
# all I did was force them to be 0,0,0 or 0,0,1 for sic always 0 or 1

update = False

if update:
    start_year=1990
    pred_year = 2018  
    end_year = pred_year - 1

    ds_79 = xr.open_mfdataset(E.obs['NSIDC_0079']['sipn_nc']+'_yearly_byweek/*byweek.nc', concat_dim='time', autoclose=True, parallel=True).sic
    ds_79=ds_79.sel(time=slice(str(start_year),str(end_year))) 

    year_all = [x.year for x in pd.to_datetime(ds_79.time.values)]
    ds_79.coords['year'] = xr.DataArray(year_all, dims='time', coords={'time':ds_79.time})

    # Write the Anomaly from the ClimoTrend for each week of each year, just use 1990-2017 fit params
    for cweek in np.arange(1,da_sic.week.max().values+1,1):

        # read in the fit parameters for this range of years
        file_in = os.path.join(mod_dir, cmod, runType, 'param_weekly', 
                                str(pred_year)+'_week'+format(cweek, '02')+'_'+str(start_year)+'_'+str(pred_year - 1)+'_SICfitparams.nc')
        ds = xr.open_mfdataset(file_in, autoclose=True, parallel=True)

        for cyear in np.arange(start_year, end_year+1, 1):
            # read in current week of current year 
            ds_specific = ds_79.where(ds_79.week==cweek, drop=True).swap_dims({'time':'year'})
            ds_specific = ds_specific.where(ds_specific.year==cyear, drop=True)

            print('Year ',ds_specific.year.values,' Week ',ds_specific.week.values)

            recons=cyear**2*ds.fitparams.isel(pdim=0,time=0) + cyear*ds.fitparams.isel(pdim=1,time=0) + ds.fitparams.isel(pdim=2,time=0)
            #x[0]**n * p[0] + ... + x[0] * p[n-1] + p[n] = y[0]
            recons.name = 'sic'
            ocnmask=recons.notnull()
            recons=recons.where(recons>0,other=0).where(ocnmask)
            recons=recons.where(recons<1,other=1).where(ocnmask)

            recons['time'].values = ds_specific['time'][0].values
            recons.values=ds_specific.values[0]-recons.values
    #        print(recons)

    #        (f, axes) = ice_plot.multi_polar_axis(ncols=4, nrows=1,sizefcter=2)
    #        recons.plot.pcolormesh(cmap='Blues',ax=axes[0], x='lon', y='lat',transform=ccrs.PlateCarree())
    #        axes[0].set_title('Week 1 Fit', fontsize=20)

            file_out = os.path.join(mod_dir, 'ObsAnomalyWeek', runType, 'sipn_nc', 
                                    str(cyear)+'_week'+format(cweek, '02')+'_SICanom.nc')
            recons.to_netcdf(file_out)
            print("Saved",file_out)


# In[ ]:


# DO NOT RUN THIS

# for cyear in np.arange(start_year, end_year+1, 1):

#    indir=os.path.join(mod_dir, 'ObsAnomalyWeek', runType, 'sipn_nc')
#    print(indir)

#    c_files = sorted(glob.glob(indir+'/'+str(cyear)+'_week'+'*.nc'))

#    ds_anom = xr.open_mfdataset(c_files, concat_dim='time')

#    file_out = os.path.join(mod_dir, 'ObsAnomalyWeek', runType, 'sipn_nc', str(cyear)+ '_SICanom.nc')

#    ds_anom.to_netcdf(file_out)
#    print("Saved",file_out)


# ### Nic's presentation figures

# In[ ]:


test_plots = False


if test_plots:
    fig_dir = '/home/disk/sipn/nicway/Nic/figures/pres/A'

    cx = 160
    cy = 220

    sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})
    f = plt.figure()
    da_train.isel(year=1).T.plot(label='Sea ice Concentration')
    plt.plot(cy,cx,marker='o',markersize=10,color='k')
    plt.title('')
    f_out = os.path.join(fig_dir,'spatial_plot.png')
    f.savefig(f_out,bbox_inches='tight', dpi=300)

    sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})

    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(da_train.isel(y=cy,x=cx).year.values, da_train.isel(y=cy,x=cx).values)
    predict_y = intercept + slope * da_train.isel(y=cy,x=cx).year.values
    predict_y

    f = plt.figure()
    da_train.isel(y=cy,x=cx).plot(color='b',label='Observed')
    plt.plot(2018, ds_pred.isel(y=cy,x=cx).values,'r*',label='Predicted',markersize=14)
    plt.plot(da_train.isel(y=cy,x=cx).year.values, predict_y,'k--', label='linear least-squares')
    plt.title('')
    plt.legend(loc='lower left', bbox_to_anchor=(1.03, .7))
    plt.ylabel('Sea Ice Concentration (-)')
    f_out = os.path.join(fig_dir,'linearfit.png')
    f.savefig(f_out,bbox_inches='tight', dpi=300)

