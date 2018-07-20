#!/bin/bash

# Example: bash run_laughter_segments.sh ../../data/transcripts/ses_list

ses_list=${1}
transcripts_dir=../../data/transcripts/
speech_seg_dir=../../data/speech_segments_chan/

while read -r line; do

  ses=$(basename ${line} '.mrt')
  mkdir -p ${speech_seg_dir}/${ses}/
  python3 create_speech_segments.py ${transcripts_dir}/${line} ${speech_seg_dir}/${ses}/

done < ${ses_list}
