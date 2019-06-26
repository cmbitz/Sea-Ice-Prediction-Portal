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
                                                                               
# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this

# Extents
#python "./plot_Extent_Model_Obs.py"
#failfunction "$?" "plot_Extent_Model_Obs.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

python "./plot_Regional_Extent.py"
#failfunction "$?" "plot_Regional_Extent.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# Maps
#python "./plot_Maps_Fast_from_database.py" 
#failfunction "$?" "plot_Maps_Fast_from_database.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# Evaluation of SIC forecasts
#python "./Eval_weekly_forecasts.py"
#failfunction "$?" "Eval_weekly_forecasts.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)"

# This needs updating
#python "./plot_Regional_maps.py"
#failfunction "$?" "plot_Regional_maps.py had an Error. See log." 

echo Finished NRT script.
