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

# Downloads data from nsidc
set -x  # Echo all lines executed
set -e  # Stop on any error

# FTP locations of data archives
# data_ftp=ftp://sidads.colorado.edu/DATASETS/nsidc0079_gsfc_bootstrap_seaice_v3/final-gsfc/north/daily/
# location moved in early 2019 to
data_ftp=https://n5eil01u.ecs.nsidc.org/PM/NSIDC-0079.003/

# Make sure the ACF Data environment variable is set
if [ -z "$NSIDC_0079_DATA_DIR" ]; then
	# Try to source path file
	echo "trying to source path_file.sh"
	source ../path_file.sh
	# Check if its now set
	if [ -z "$NSIDC_0079_DATA_DIR" ]; then
		echo "Need to set NSIDC_0079_DATA_DIR"
		exit 1
	fi
fi

mkdir -p $NSIDC_0079_DATA_DIR

# Download
cd $NSIDC_0079_DATA_DIR

#wget  -r --no-parent -A 'bt_2018*n.bin' ${data_ftp}
#wget -nH --cut-dirs=20 -r -A n.bin -N $data_ftp
#wget -nH --cut-dirs=20 -r -A bt_20181231_f17_v3.1_n.bin -N $data_ftp

# these work
y=2018
for doy in {0..365}
	   do

dd=`date -d "${doy} days 2018-01-01" +"%Y%m%d"`
day=${dd:6:2}
mon=${dd:4:2} 
echo date $dd  month $mon day $day

wget ${data_ftp}/${y}.${mon}.${day}/bt_${dd}_f17_v3.1_n.bin

done

echo "Done!"

