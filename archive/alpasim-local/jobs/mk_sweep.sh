#!/bin/bash
# usage: mk_sweep.sh <tag> <ALON> <ALAT>
T=$1; ALON=$2; ALAT=$3
sed -e "s|RF_V_MAX=[0-9]* python|RF_V_MAX=20 RF_A_LON=$ALON RF_A_LAT=$ALAT python|" \
    -e "s|python \$REPO/../alpasim-challenge/driver/route_follower.py|RF_V_MAX=20 RF_A_LON=$ALON RF_A_LAT=$ALAT python \$REPO/../alpasim-challenge/driver/route_follower.py|" \
    -e "s|rfv16-|$T-|g; s|rf9-|$T-|g" \
    -e "s|nuplan-rfv16|nuplan-$T|; s|nuplan-rf9|nuplan-$T|" \
    ~/alpasim/jobs/rfv16.sbatch > ~/alpasim/jobs/$T.sbatch
echo "--- $T ---"; grep -n "RF_A_LON.*route_follower" ~/alpasim/jobs/$T.sbatch
