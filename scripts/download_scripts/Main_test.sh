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
#source $HOME/.bashrc
source /home/disk/sipn/nicway/.bashrc
source ../path_file.sh
source activate esio


failfunction()
{
    if [ "$1" != 0 ]
    then echo "One of the commands has failed! Mailing for help."
        mail -s "Error in Daily SIPN2 run."  $EMAIL <<< $2
	exit
    fi
}

# testing failfunction
#cd /home/asdfsd/ 
#failfunction "$?" "test failed"

# Make sure the ACF REPO_DIR environment variable is set
if [ -z "$REPO_DIR" ]; then
    echo "Need to set REPO_DIR"
    exit 1
fi

# Model downloads
                                                                               
#python $REPO_DIR"/scripts/download_scripts/Download_s2s.py" "recent" 
#failfunction "$?" "Download_s2s.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# python $REPO_DIR"/scripts/download_scripts/Download_C3S.py" "recent" 
# Allowing fail of ukmo and ecmwf for now
#failfunction "$?" "Download_C3S.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

#$REPO_DIR"/scripts/download_scripts/download_RASM_ESRL.sh" 
#failfunction "$?" "download_RASM_ESRL.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)."

#$REPO_DIR"/scripts/download_scripts/download_NRL_GOFS3_1.sh"
#failfunction "$?" "download_NRL_GOFS3_1.sh had an Error. See log. (https://atmos.washington.edu/~nicway/sipn/log/)"

#wait # Below depends on above

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this

# Import Models to sipn format
#source activate pynioNew # Required to process grib files
#python "./Regrid_S2S_Models.py"
#failfunction "$?" "Regrid_S2S_Models.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

#python "./Regrid_RASM.py"
#failfunction "$?" "Regrid_RASM.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

#python "./Regrid_CFSv2.py"
#failfunction "$?" "Regrid_CFSv2.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

wait
source activate esio
wait # Below depends on above

echo Aggregations
# Calc Aggregate metrics (e.g. extent for different regions)
#python "./Calc_Model_Aggregations.py"
#failfunction "$?" "Calc_Model_Aggregations.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

echo "Damped Persistence"
# Aggregate to weekly mean, anomaly, SIP
#python "./Model_Damped_Anomaly_Persistence.py"
#failfunction "$?" "Model_Damped_Anomaly_Persistence.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)."

echo "To Weekly"
# Aggregate SIC to weekly forecasts
#python "./Calc_Weekly_Model_Metrics.py"
#failfunction "$?" "Calc_Weekly_Model_Metrics.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)."

echo "Weekly to Zarr"
# Aggregate Monthly CHunked Zarr files to one big Zarr file
#python "./Agg_Weekly_to_Zarr.py"
#failfunction "$?" "Agg_Weekly_to_Zarr.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)."

echo "Zarr to Cloud"
# Upload Zarr files to Google Cloud Bucket
#/home/disk/sipn/nicway/data/model/zarr/upload.sh

echo "python plots"
# Make Plots
which python

echo "plot extent"
# Extents
python "./plot_Extent_Model_Obs.py"
#failfunction "$?" "plot_Extent_Model_Obs.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

echo "plot regional extent"
python "./plot_Regional_Extent.py"
#failfunction "$?" "plot_Regional_Extent.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

echo "plot maps"
# Maps
python "./plot_Maps_Fast_from_database.py" 
#failfunction "$?" "plot_Maps_Fast_from_database.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

echo "evaluate SIC"
# Evaluation of SIC forecasts
python "./Eval_weekly_forecasts.py"
#failfunction "$?" "Eval_weekly_forecasts.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)"

# This needs updating
#python "./plot_Regional_maps.py"
#failfunction "$?" "plot_Regional_maps.py had an Error. See log." 

echo Finished NRT script.
