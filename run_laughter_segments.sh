#!/bin/bash

# Example: sudo bash run_laughter_segments.sh ../../data/ses_list

ses_list=${1}
transcripts_dir=../../data/transcripts/
laughter_stats_dir=../../data/laughter_segments_chan/

while read -r ses; do

  mkdir -p ${laughter_stats_dir}/${ses}/
  python3 create_laughter_segments.py ${transcripts_dir}/${ses}.mrt ${laughter_stats_dir}/${ses}/

done < ${ses_list}
