
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

from ecmwfapi import ECMWFDataServer
import os
import sys
import dask
import numpy as np
from esio import download
import datetime
from esio import EsioData as ed
import pandas as pd

import calendar
import datetime

import os
from ecmwfapi import ECMWFDataServer

'''
Download models with sea ice forecasts within the 2s2 forecast data archive.

https://software.ecmwf.int/wiki/display/S2S/Models

'''


# In[22]:


# ukmo makes a hindcast everytime they make a new forecast
# they label these hindcast by the Model Version Date
# which oddly is not the current date but rather a few weeks ahead
# Then for eace of these versions, they run a forecast for the same mon and day
# from 1993 to one year ago

# the MVDs are on 1, 9, 17, 25 
dataclass='s2s'
run_type='reforecast'
cmod = 'ukmo'

# end on last day of previous month for convenience, doesn't really matter 
# since I assume the model is not THAT different over a month or two
cd = datetime.datetime.now()
DS = datetime.datetime(cd.year-1, cd.month, 1) # 
DE = datetime.datetime(cd.year, cd.month-1, 28) 

earlyinmonth = pd.date_range(start=DS,end=DE,freq='SMS-9')  # 1st and 9th of month
laterinmonth = pd.date_range(start=DS,end=DE,freq='SMS-17')  # 1st and 17th of month
evenlaterinmonth = pd.date_range(start=DS,end=DE,freq='SMS-25')  # 1st and 25th of month
alldates=np.append(earlyinmonth,laterinmonth)               #smoosh together
versions=np.unique(np.append(alldates,evenlaterinmonth))    #sort and ditch duplicates

vdates = pd.DatetimeIndex(versions)
print(vdates) # glad that worked
len(vdates)


# In[23]:


vdates=vdates[14:15]
print(vdates)
len(vdates)

#exit()

# In[24]:


E = ed.EsioData.load()
main_dir = E.model_dir


# In[25]:


# a starting point
mod_dicts = {}

# UKMO
mod_dicts['ukmo'] = {
 "class": "s2",
 "dataset": "s2s",
 "date": "2019-07-11",                                                               
 "hdate": "2005-07-11",                                                          
 "expver": "prod",
 "levtype": "sfc",
 "model": "glob",
 "origin": "egrr",
 "param": "31",
 "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440",
 "stream": "enfh",
 "time": "00:00:00",
 "number": "1/2/3/4/5/6",
 "type": "pf",
 "target": "output",
 }


# In[26]:


@dask.delayed

def download_month(config_dict):
    # Start server
    cserver = ECMWFDataServer()
    cserver.retrieve(config_dict)
    return 1


# In[27]:


# missing data in year 2016 for unknown reason


# In[28]:



for ct in np.arange(0,len(vdates),1):
    X = 1

    year = vdates[ct].year
    month = vdates[ct].month
    day = vdates[ct].day
    print(year, month, day)
    DS = np.datetime64(datetime.datetime(1995, month, day, 0, 0))
    DE = np.datetime64(datetime.datetime(2015, month, day, 0, 0))
    
    hdates = pd.date_range(start=DS,end=DE,freq=pd.DateOffset(years=1))
    hdates = [x.strftime('%Y-%m-%d') for x in hdates]
    vdate = vdates[ct].strftime('%Y-%m-%d')
    print(hdates)
    
    for hd in np.arange(0,len(hdates),1):

        cdict = mod_dicts[cmod]
        cdict['date'] = vdate
        cdict['hdate'] = hdates[hd]

        target = os.path.join(main_dir, cmod, run_type, 'native',
                                  cmod+'_'+hdates[hd]+'.grib')
        cdict['target'] = target
        cdict['expect'] = 'any'

        print('Version Date is ',vdate,' hdate is ',hdates[hd])
        print('output file is ',target)
        print('/n/n')
        #X = download_month(cdict)
        #X.compute()
        X = X + download_month(cdict)

    # Call compute to download all years concurently for this date
    X.compute()


