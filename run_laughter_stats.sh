#!/bin/bash

transcripts_dir=/home/raghuveer/work/TILES/work_in_progress/data/transcripts/
laughter_stats_dir=/home/raghuveer/work/TILES/work_in_progress/data/laughter_stats/

while read -r line; do

ses=$(basename ${line} '.mrt')
mkdir -p ${laughter_stats_dir}/${ses}/

python3 get_laughter_stats.py ${transcripts_dir}/${line} ${laughter_stats_dir}/${ses}/

done < ${1}
