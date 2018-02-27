#!/bin/bash

set -e
cd ~/gpdb
mkdir -p gpconfigs
source /usr/local/gpdb/greenplum_path.sh
cp $GPHOME/docs/cli_help/gpconfigs/gpinitsystem_config \
    ~/gpdb/gpconfigs/gpinitsystem_config
cd gpconfigs
sudo ldconfig

export me=`hostname`
sed -i s/MASTER_HOSTNAME=mdw/MASTER_HOSTNAME=${me}/g gpinitsystem_config
echo "${me}" >> hostfile_gpinitsystem

cd ../
yes | gpinitsystem -c gpconfigs/gpinitsystem_config -h \
    gpconfigs/hostfile_gpinitsystem
