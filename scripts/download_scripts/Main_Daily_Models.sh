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

# This script automatically downloads model data and regrids it
# computes extents and plots them each day with the obs

# Set up python paths
source $HOME/.bashrc
#source /home/disk/sipn/nicway/.bashrc
source /home/disk/sipn/bitz/python/ESIO/scripts/path_file.sh
source activate esio
export PYTHONPATH="/home/disk/sipn/bitz/python/ESIO":$PTHONPATH                                                


failfunction()
{
    if [ "$1" != 0 ]
    then echo "One of the commands has failed! NOT Mailing for help."
       # mail -s "Error in Daily SIPN2 run."  $EMAIL <<< $2
       # exit
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
# this is working fine

python $REPO_DIR"/scripts/download_scripts/Download_s2s.py" "recent" 
#failfunction "$?" "Download_s2s.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# GET NRL NESM model ~45 day lead times
# put into $DATA_DIR/model/usnavyncep  last downloaded on 5/1/2019 when I checked, so working
# this script seems to stop unless it downloads something
echo "Downloading NRL"
$REPO_DIR"/scripts/download_scripts/download_NRL.sh"
failfunction "$?" "download_NRL.sh had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# Get NRL SIO NESM model longer lead times
# put into $DATA_DIR/model/usnavysipn no data since 8/15/2018 but appears to check
# correctly for new data
echo "Downloading NRL SIO"
$REPO_DIR"/scripts/download_scripts/download_NRL_SIO.sh"
failfunction "$?" "download_NRL_SIO.sh had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 



##### BEWARE NEEDS ATTENTION SHORTLY see sh script
$REPO_DIR"/scripts/download_scripts/download_RASM_ESRL.sh" 
#failfunction "$?" "download_RASM_ESRL.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)."

$REPO_DIR"/scripts/download_scripts/download_NRL_GOFS3_1.sh"
#failfunction "$?" "download_NRL_GOFS3_1.sh had an Error. See log. (https://atmos.washington.edu/~nicway/sipn/log/)"


# CC modification: refresh the html numbers appended to figures to force reload
# trouble is that some browswers cache too aggressively
# there is probabaly a better way to do this, but it seems to be working
cd /home/disk/sipn/nicway/public_html/sipn
/home/disk/sipn/nicway/public_html/sipn/chngnum.sh

wait # Below depends on above

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this


# Import Models to sipn format
echo "Regrid NESM"
python "./Regrid_NESM.py"
failfunction "$?" "Regrid_NESM.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# GFDL (monthly) this data appears to be uploaded by gfdl only during summer so stop for now
echo "Regrid GFDL"
#python "./Regrid_GFDL_Forecast.py"
#failfunction "$?" "Regrid_GFDL_Forecast.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

echo "regridding FGOALS"
# must figure out how to make more robust since they periodically change their data format
python "./Regrid_FGOALS_Forecast.py"
#failfunction "$?" "Regrid_FGOALS_Forecast.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 


# Import S2S Models and regrid to sipn format
source activate pynioNew # Required to process grib files
export PYTHONPATH="/home/disk/sipn/bitz/python/ESIO":$PTHONPATH                                                

python "./Regrid_S2S_Models.py" "S2S"
failfunction "$?" "Regrid_S2S_Models.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

python "./Regrid_RASM.py"
#failfunction "$?" "Regrid_RASM.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

python "./Regrid_CFSv2.py"
failfunction "$?" "Regrid_CFSv2.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

wait

# the day of week is
DOW=$(date +%u)

if [[ $DOW == 2 ]]; then
    echo It is Tuesday so skip rest of the script
    echo Finished NRT script.
    exit
fi

source activate esio
export PYTHONPATH="/home/disk/sipn/bitz/python/ESIO":$PTHONPATH                                                


# Calc Regional extents on daily data (includes ClimoTrend and DampAnom)
python "./Calc_Model_Aggregations.py"
failfunction "$?" "Calc_Model_Aggregations.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)."


# this is CC's major overhaul of this script
echo "Running plot_Regional_Extent_TwoPanel.py"
python "./plot_Regional_Extent_TwoPanel.py"
failfunction "$?" "plot_Regional_Extent_TwoPanel.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 


echo Finished NRT script.
