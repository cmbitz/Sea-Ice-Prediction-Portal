#!/bin/bash

#This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
#If you use this code for a publication or presentation, please cite the reference in the README.md on the
#main page (https://github.com/NicWayand/ESIO). 
#
#Questions or comments should be addressed to nicway@uw.edu
#
#Copyright (c) 2018 Nic Wayand
#
#GNU General Public License v3.0

#set -x  # Echo all lines executed
#set -e  # Stop on any error

# Set up python paths
source $HOME/.bashrc
#source /home/disk/sipn/nicway/.bashrc  
source /home/disk/sipn/bitz/python/ESIO/scripts/path_file.sh
source activate esio
export PYTHONPATH="/home/disk/sipn/bitz/python/ESIO":$PTHONPATH                                                

which python

failfunction()
{
    if [ "$1" != 0 ]
    then echo "One of the commands has failed! NOT Mailing for help."
#        mail -s "Error in Daily SIPN2 run." $EMAIL <<< $2
	exit
    fi
}

# Make sure the ACF REPO_DIR environment variable is set
if [ -z "$REPO_DIR" ]; then
    echo "Need to set REPO_DIR"
    exit 1
fi

# Call all download scripts that grab near-real-time data
$REPO_DIR"/scripts/download_scripts/download_NSIDC_0081.sh" & # Fast
$REPO_DIR"/scripts/download_scripts/download_NSIDC_extents.sh"  # Fast
echo "done with NSIDC downloads"
failfunction "$?" "download_NSIDC_0081.sh or download_NSIDC_extents.sh had an Error. See log." 

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this

python "./Agg_NSIDC_Obs.py"
failfunction "$?" "Agg_NSIDC_Obs.py had an Error. See log." 

echo "Main_6hrly: done major analysis on obs in Agg_NSIDC_Obs"

# ClimoTrend of the weekly means, takes 5-10 mins usually, depends on previous 
python "./Calc_Obs_ClimoTrendWeekly.py"
failfunction "$?" "Agg_NSIDC_Obs.py had an Error. See log." 

echo "Main_6hrly: done with Calc_Obs_ClimoTrendWeekly"

# ClimoTrend of the weekly means, takes 5-10 mins usually, depends on previous
python "./Interpolate_ClimoTrend_weekly_to_daily.py"
failfunction "$?" "Interpolate_ClimoTrend_weekly_to_daily.py had an Error. See log." 

echo "Main_6hrly: done Interpolate_ClimoTrend_weekly_to_daily"

#  run for fun,  not being used for anything further. it makes a nice fig
python "./Calc_Obs_DampAnomWeekly.py"
failfunction "$?" "Calc_Obs_DampanomWeekly.py had an Error. See log." 
#  Make the daily damped anomaly benchmark, takes a few minutes
python "./Calc_Obs_DampAnomDaily.py"
failfunction "$?" "Calc_Obs_DampAnomDaily.py had an Error. See log."

echo "Main_6hrly: done with DampAnom stuff"

# CC thinks this is working perhaps should modify to put the regional extents etc on clouds
# Convert obs only to Zarr, not sure why exactly
python "./Convert_netcdf_to_Zarr.py"
failfunction "$?" "Convert_netcdf_to_Zarr.py had an Error. See log."

# CC is not working cuz gsutil is not in my path, not sure where it is
# Upload to GCP
# /home/disk/sipn/nicway/data/obs/zarr/update_obs.sh

# Import Models to sipn format
source activate pynioNew # Requires new env
python "./Regrid_YOPP.py"
failfunction "$?" "Regrid_YOPP.py had an Error. See log." 

source activate esio

wait # Below depends on above

# Make Plots

# Observations
echo "plot observations"
python "./plot_observations.py" 
failfunction "$?" "plot_observations.py had an Error. See log." 

# Availblity plots, giving memory errors now so skip until have time to fix
#echo "plot forecast availability"
#python "./plot_forecast_availability.py" &
#failfunction "$?" "plot_forecast_availability.py had an Error. See log." 


echo Finished NRT script but python still running to make plots.
