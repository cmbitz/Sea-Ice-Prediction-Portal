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
set -e  # Stop on any error

# Set up python paths
source $HOME/.bashrc
#source /home/disk/sipn/nicway/.bashrc
source /home/disk/sipn/bitz/python/ESIO/scripts/path_file.sh
source activate esio
which python

# Make sure the ACF REPO_DIR environment variable is set
if [ -z "$REPO_DIR" ]; then
    echo "Need to set REPO_DIR"
    exit 1
fi

failfunction()
{
    if [ "$1" != 0 ]
    then echo "One of the commands has failed! NOT Mailing for help."
#        mail -s "Error in Daily SIPN2 run."  $EMAIL <<< $2
	exit
    fi
}

# GET NRL NESM model ~45 day lead times
# put into $DATA_DIR/model/usnavyncep  last downloaded on 5/1/2019 when I checked, so working
$REPO_DIR"/scripts/download_scripts/download_NRL.sh"
failfunction "$?" "download_NRL.sh had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# Get NRL SIO NESM model longer lead times
# put into $DATA_DIR/model/usnavysipn no data since 8/15/2018 but appears to check
# correctly for new data
$REPO_DIR"/scripts/download_scripts/download_NRL_SIO.sh"
failfunction "$?" "download_NRL_SIO.sh had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# Regrid them
wait # Below depends on above

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this

# Import Models to sipn format
python "./Regrid_NESM.py"
failfunction "$?" "Regrid_NESM.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# GFDL (monthly) this data appears to be uploaded by gfdl 
python "./Regrid_GFDL_Forecast.py"
failfunction "$?" "Regrid_GFDL_Forecast.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

echo Finished Weekly script.


