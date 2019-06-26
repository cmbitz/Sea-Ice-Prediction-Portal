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
source /home/disk/sipn/nicway/python/ESIO/scripts/path_file.sh
source activate esio


failfunction()
{
    if [ "$1" != 0 ]
    then echo "One of the commands has failed! NOT Mailing for help."
       # mail -s "Error in Daily SIPN2 run."  $EMAIL <<< $2
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


# CC modification: refresh the html numbers appended to figures to force reload
# trouble is that some browswers cache too aggressively
# there is probabaly a better way to do this, but it seems to be working
cd /home/disk/sipn/nicway/public_html/sipn
/home/disk/sipn/nicway/public_html/sipn/chngnum.sh

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this


wait
source activate esio
wait # Below depends on above



# Aggregate SIC to weekly forecasts 
python "./Calc_Weekly_Model_Metrics.py"
failfunction "$?" "Calc_Weekly_Model_Metrics.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)."


# Aggregate Monthly CHunked Zarr files to one big Zarr file
python "./Agg_Weekly_to_Zarr.py"
#failfunction "$?" "Agg_Weekly_to_Zarr.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)."

# CC is not working cuz gsutil is not in my path, not sure where it is
# Upload Zarr files to Google Cloud Bucket
 /home/disk/sipn/nicway/data/model/zarr/upload.sh

# Make Plots
which python

# Extents
# CC modification: merged total extent plot into regional extent script so do not call
#python "./plot_Extent_Model_Obs.py"
#failfunction "$?" "plot_Extent_Model_Obs.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# this is CC's major overhaul of this script
python "./plot_Regional_Extent.py"
failfunction "$?" "plot_Regional_Extent.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# Maps takes about 7-10 min to make one init_times set of figures (each has 10 lead times x 3 metrics)
python "./plot_Maps_Fast_from_database.py" 
failfunction "$?" "plot_Maps_Fast_from_database.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# Evaluation of SIC forecasts
python "./Eval_weekly_forecasts.py"
failfunction "$?" "Eval_weekly_forecasts.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)"

# This needs updating, Nic never got it going
#python "./plot_Regional_maps.py"
#failfunction "$?" "plot_Regional_maps.py had an Error. See log." 

echo Finished NRT script.
