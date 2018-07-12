#!/bin/bash

# Example: bash run_laughter_stats.sh ../../data/transcripts/ses_list

transcripts_dir=../../data/transcripts/
laughter_stats_dir=../../data/laughter_segments/

while read -r line; do

  ses=$(basename ${line} '.mrt')
  mkdir -p ${laughter_stats_dir}/${ses}/
  python3 get_laughter_stats.py ${transcripts_dir}/${line} ${laughter_stats_dir}/${ses}/

done < ${1}
