import calendar
import datetime
import os
import dask
import numpy as np
import pandas as pd
import cdsapi


@dask.delayed
def download_month(datatype,config_dict,target):
    # Start server
    cserver = cdsapi.Client()
    print("Requesting data with datatype ",datatype)
    print("configuration ",config_dict)
    print("and target ",target)
    cserver.retrieve(datatype,config_dict,target)
    return 1

def download_data_by_month(dataclass=None, datatype=None, main_dir=None,
                           mod_dicts=None, cy=None, cm=None,
                           run_type='forecast'):
    X = 1

    if dataclass=='s2s':
        day_lag = 22
    elif dataclass=='c3':
        day_lag = 16   
    else:
        raise ValueError('dataclass not found.')

    DS = datetime.datetime(cy,cm,1)
    DE = datetime.datetime(cy,cm,calendar.monthrange(cy,cm)[1])

    cd = datetime.datetime.now()
    S2S = cd - datetime.timedelta(days=day_lag)

    # Check if current month, insure dates are not within last 16/21 days (embargo)
    DE = np.min([DE, S2S])

    # Check if most recent init date is before current month start
    if DS>DE:
        print('No data available yet for ', str(cy),'-',str(cm))
        print('Re-downloading previous month...')
        cm = cm -1
        if cm==0:
            cm =  12
            cy = cy - 1
        print(cm)
        download_data_by_month(dataclass=dataclass, datatype=datatype, main_dir=main_dir,
                               mod_dicts=mod_dicts, cy=cy, cm=cm)
        return 0 # Just return an int for dask. Don't continue here.

    for cmod in mod_dicts.keys():
        print(cmod)
        cdict = mod_dicts[cmod]
        # Update cdict
        cdict['year'] = str(cy)
        cdict['month'] = format(cm, '02')
        target = os.path.join(main_dir, cmod, run_type, 'native',
                              cmod+'_'+str(cy)+'_'+format(cm, '02')+'.grib')
        X = X + download_month(datatype,cdict,target)

    # Call compute to download all models concurently from ecmwf
    X.compute()
