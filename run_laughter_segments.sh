#!/bin/bash

# Example: bash run_laughter_stats.sh ../../data/transcripts/ses_list

ses_list=${1}
transcripts_dir=../../data/transcripts/
laughter_stats_dir=../../data/laughter_segments_chan/

while read -r line; do

  ses=$(basename ${line} '.mrt')
  mkdir -p ${laughter_stats_dir}/${ses}/
  python3 create_laughter_segments.py ${transcripts_dir}/${line} ${laughter_stats_dir}/${ses}/

done < ${ses_list}
