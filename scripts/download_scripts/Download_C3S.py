
# coding: utf-8

# In[11]:


'''

This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
If you use this code for a publication or presentation, please cite the reference in the README.md on the
main page (https://github.com/NicWayand/ESIO). 

Questions or comments should be addressed to nicway@uw.edu

Copyright (c) 2018 Nic Wayand

GNU General Public License v3.0


'''
import cdsapi
import os
import sys
import dask
import numpy as np
import datetime
import calendar

'''
Download models with sea ice forecasts from the Climate Data Service

https://cds.climate.copernicus.eu/

'''


# In[3]:


# do not run this part when from Jupyter notebook, instead run next cell

# Check user defined configuraiton file
if len(sys.argv) < 2:
    raise ValueError('Requires either one arguments [recent] \n or two [start_year, end_yaer] [start_month, end_month] (inclusive) ')

# Get name of configuration file/module
timeperiod = sys.argv[1]
if timeperiod=='recent':
    cd = datetime.datetime.now()
    years = cd.year
    months = cd.month
else:
    year_range_in = list(map(int, sys.argv[1].strip('[]').split(',')))
    month_range_in = list(map(int, sys.argv[2].strip('[]').split(',')))
    if (len(year_range_in)!=2) | (len(month_range_in)!=2):
        raise ValueError('Year range and month range must be two values (inclusive)')
    years = np.arange(year_range_in[0], year_range_in[1]+1, 1)
    months = np.arange(month_range_in[0], month_range_in[1]+1, 1)
    assert np.all(months>=0), 'months must be >=0'
    assert np.all(months<=12), 'months must be <=12'


# In[59]:


# Testing
#years = 2004
#years=np.arange(2000,2019,1)
#months = np.arange(1,7,1)

# OR MOST RECENT
#cd = datetime.datetime.now()
#years = cd.year
#months = cd.month
print(cd, years, months)


# In[13]:


@dask.delayed
def download_month(datatype,config_dict,target):
    # Start server
    cserver = cdsapi.Client()
    print("Requesting data with datatype ",datatype)
    print("configuration ",config_dict)
    print("and target ",target)
    cserver.retrieve(datatype,config_dict,target)
    return 1


# In[14]:


main_dir = '/home/disk/sipn/nicway/data/model'


# In[85]:


# Templet dicts for each model
### ukmetoffice is upgrading to system 14 at the moment only some months are done for retrospective
# so get system 13 and 14 both
# reforecast ukmetoffice only made on 01, 09, 17, 25

# Init it
mod_dicts = {}

mod_dicts['ecmwfsipn']  =    {'format':'grib',
     'originating_centre':'ecmwf',
     'system':'5',
     'variable':'sea_ice_cover',
     'year':'placeholder',
     'month':'placeholder',
     'day':'01',
     'leadtime_hour':['24','48','72',
            '96','120','144',
            '168','192','216',
            '240','264','288',
            '312','336','360',
            '384','408','432',
            '456','480','504',
            '528','552','576',
            '600','624','648',
            '672','696','720',
            '744','768','792',
            '816','840','864',
            '888','912','936',
            '960','984','1008',
            '1032','1056','1080',
            '1104','1128','1152',
            '1176','1200','1224',
            '1248','1272','1296',
            '1320','1344','1368',
            '1392','1416','1440',
            '1464','1488','1512',
            '1536','1560','1584',
            '1608','1632','1656',
            '1680','1704','1728',
            '1752','1776','1800',
            '1824','1848','1872',
            '1896','1920','1944',
            '1968','1992','2016',
            '2040','2064','2088',
            '2112','2136','2160',
            '2184','2208','2232',
            '2256','2280','2304',
            '2328','2352','2376',
            '2400','2424','2448',
            '2472','2496','2520',
            '2544','2568','2592',
            '2616','2640','2664',
            '2688','2712','2736',
            '2760','2784','2808',
            '2832','2856','2880',
            '2904','2928','2952',
            '2976','3000','3024',
            '3048','3072','3096',
            '3120','3144','3168',
            '3192','3216','3240',
            '3264','3288','3312',
            '3336','3360','3384',
            '3408','3432','3456',
            '3480','3504','3528',
            '3552','3576','3600',
            '3624','3648','3672',
            '3696','3720','3744',
            '3768','3792','3816',
            '3840','3864','3888',
            '3912','3936','3960',
            '3984','4008','4032',
            '4056','4080','4104',
            '4128','4152','4176',
            '4200','4224','4248',
            '4272','4296','4320',
            '4344','4368','4392',
            '4416','4440','4464',
            '4488','4512','4536',
            '4560','4584','4608',
            '4632','4656','4680',
            '4704','4728','4752',
            '4776','4800','4824',
            '4848','4872','4896',
            '4920','4944','4968',
            '4992','5016','5040',
            '5064','5088','5112',
            '5136','5160'],
             'format':'grib'      }

# ukmetofficesipn only provides 01 until much later, not sure when the the rest
# of the days become available but for sure by the 20th
# so get only 01 for the first and then go back and get previous month for 
# the whole set

mod_dicts['ukmetofficesipn']  =    {'format':'grib',
     'originating_centre':'ukmo',
     'system':'14',
     'variable':'sea_ice_cover',
     'year':'placeholder',
     'month':'placeholder',
     'day':['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'],

     'leadtime_hour':['24','48','72','96','120','144','168','192','216','240','264','288','312','336','360','384','408','432','456','480','504','528','552','576','600','624','648','672','696','720','744','768','792','816','840','864','888','912','936','960','984','1008','1032','1056','1080','1104','1128','1152','1176','1200','1224','1248','1272','1296','1320','1344','1368','1392','1416','1440','1464','1488','1512','1536','1560','1584','1608','1632','1656','1680','1704','1728','1752','1776','1800','1824','1848','1872','1896','1920','1944','1968','1992','2016','2040','2064','2088','2112','2136','2160','2184','2208','2232','2256','2280','2304','2328','2352','2376','2400','2424','2448','2472','2496','2520','2544','2568','2592','2616','2640','2664','2688','2712','2736','2760','2784','2808','2832','2856','2880','2904','2928','2952','2976','3000','3024','3048','3072','3096','3120','3144','3168','3192','3216','3240','3264','3288','3312','3336','3360','3384','3408','3432','3456','3480','3504','3528','3552','3576','3600','3624','3648','3672','3696','3720','3744','3768','3792','3816','3840','3864','3888','3912','3936','3960','3984','4008','4032','4056','4080','4104','4128','4152','4176','4200','4224','4248','4272','4296','4320','4344','4368','4392','4416','4440','4464','4488','4512','4536','4560','4584','4608','4632','4656','4680','4704','4728','4752','4776','4800','4824','4848','4872','4896','4920','4944','4968','4992','5016','5040','5064','5088','5112','5136','5160'],
             'format':'grib'      }


mod_dicts['meteofrsipn']  =    {'format':'grib',
     'originating_centre':'meteo_france',
     'system':'',
     'variable':'sea_ice_cover',
     'year':'placeholder',
     'month':'placeholder',
     'day':'01',
     'leadtime_hour':['24','48','72',
            '96','120','144',
            '168','192','216',
            '240','264','288',
            '312','336','360',
            '384','408','432',
            '456','480','504',
            '528','552','576',
            '600','624','648',
            '672','696','720',
            '744','768','792',
            '816','840','864',
            '888','912','936',
            '960','984','1008',
            '1032','1056','1080',
            '1104','1128','1152',
            '1176','1200','1224',
            '1248','1272','1296',
            '1320','1344','1368',
            '1392','1416','1440',
            '1464','1488','1512',
            '1536','1560','1584',
            '1608','1632','1656',
            '1680','1704','1728',
            '1752','1776','1800',
            '1824','1848','1872',
            '1896','1920','1944',
            '1968','1992','2016',
            '2040','2064','2088',
            '2112','2136','2160',
            '2184','2208','2232',
            '2256','2280','2304',
            '2328','2352','2376',
            '2400','2424','2448',
            '2472','2496','2520',
            '2544','2568','2592',
            '2616','2640','2664',
            '2688','2712','2736',
            '2760','2784','2808',
            '2832','2856','2880',
            '2904','2928','2952',
            '2976','3000','3024',
            '3048','3072','3096',
            '3120','3144','3168',
            '3192','3216','3240',
            '3264','3288','3312',
            '3336','3360','3384',
            '3408','3432','3456',
            '3480','3504','3528',
            '3552','3576','3600',
            '3624','3648','3672',
            '3696','3720','3744',
            '3768','3792','3816',
            '3840','3864','3888',
            '3912','3936','3960',
            '3984','4008','4032',
            '4056','4080','4104',
            '4128','4152','4176',
            '4200','4224','4248',
            '4272','4296','4320',
            '4344','4368','4392',
            '4416','4440','4464',
            '4488','4512','4536',
            '4560','4584','4608',
            '4632','4656','4680',
            '4704','4728','4752',
            '4776','4800','4824',
            '4848','4872','4896',
            '4920','4944','4968',
            '4992','5016','5040',
            '5064'
        ]
 
                               
}

#     'day':['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31'],


# In[86]:


mod_dicts.keys()


# In[87]:





# In[88]:


# check if this month, if so then get just day 01
if ((months==cd.month) and (years==cd.year)):
    mod_dicts['ukmetofficesipn']['day'] = '01'


# In[90]:





# In[89]:


datatype='seasonal-original-single-levels'
run_type='forecast'
X=1

monthstr = np.char.mod('%02d', months).tolist()

for cmod in mod_dicts.keys():
    cdict = mod_dicts[cmod]
    cdict['year'] = str(years)
    cdict['month'] = monthstr

    target = os.path.join(main_dir, cmod, run_type,'native',
                  cmod+'_'+str(years)+'_'+monthstr+'.grib')

    X = X + download_month(datatype,cdict,target)


# check if this month and if so get ukmet for previous months all days too
if ((months==cd.month) and (years==cd.year)):
    cmod = 'ukmetofficesipn'
    mod_dicts[cmod]['day'] = ['01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23','24','25','26','27','28','29','30','31']
    months = months-1
    if months == 0:
        months = 12
        years = years -1
    monthstr = np.char.mod('%02d', months).tolist()
    cy=str(years)

    cdict = mod_dicts[cmod]
    cdict['year'] = str(years)
    cdict['month'] = monthstr

    target = os.path.join(main_dir, cmod, run_type,'native',
                  cmod+'_'+str(years)+'_'+monthstr+'.grib')

    X = X + download_month(datatype,cdict,target)
    
X.compute()


# In[91]:




