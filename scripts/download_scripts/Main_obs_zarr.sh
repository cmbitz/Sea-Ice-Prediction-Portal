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




############

# CC is not sure if this is working
# error is /home/disk/sipn/nicway/data/obs/zarr/update_obs.sh: line 7: ./upload.sh: No such file or directory
# Upload to GCP
/home/disk/sipn/nicway/data/obs/zarr/update_obs.sh

