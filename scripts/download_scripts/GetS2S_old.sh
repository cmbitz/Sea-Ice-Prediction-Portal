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
source ../path_file.sh
source activate esio


failfunction()
{
    if [ "$1" != 0 ]
    then echo "One of the commands has failed! NOT Mailing for help."
#        mail -s "Error in Daily SIPN2 run."  $EMAIL <<< $2
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
                                                                               
python $REPO_DIR"/scripts/download_scripts/Download_s2s.py" "[2005,2005]" "[1,2]"
failfunction "$?" "Download_s2s.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# python $REPO_DIR"/scripts/download_scripts/Download_C3S.py" 
# Allowing fail of ukmo and ecmwf for now
#failfunction "$?" "Download_C3S.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

wait # Below depends on above

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this

# Import Models to sipn format
source activate pynioNew # Required to process grib files
python "./Regrid_S2S_Models.py"
#failfunction "$?" "Regrid_S2S_Models.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 


echo Finished NRT script.
