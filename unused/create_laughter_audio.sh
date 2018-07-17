#!/bin/bash

# This script takes as input a laughter segments file, input laughter wav directory and creates laughter segments audio

typeOfSeg=${1}  #breath_laugh #laugh_utterance #while_laughing #laugh_only

ses_file=/home/raghuveer/work/TILES/laughter/data/ses_list
seg_dir=/home/raghuveer/work/TILES/laughter/data/laughter_segments/
inp_wav_dir=/media/External_HD/tiles_audio/icsi_close_mic_wav/
output_wav_dir=/media/External_HD/tiles_audio/icsi_laughter_wav_segments/${typeOfSeg}/
#/home/raghuveer/work/TILES/icsi_laughter_wav_segments/${typeOfSeg}
#/media/External_HD/tiles_audio/icsi_laughter_wav_segments/${typeOfSeg}/

while read -r ses; do
  out_dir=${output_wav_dir}/${ses}/
  mkdir -p ${out_dir}

  seg_file=${seg_dir}/${ses}/${typeOfSeg}_segments.txt
  while read -r line; do
    channel=$(echo $line | cut -d" " -f 3)

    start=$(echo $line | cut -d" " -f 1)
    end=$(echo $line | cut -d" " -f 2)

    dur=$(echo "$end - $start" | bc)

    st=$(echo ${start} | cut -d"." -f1)_$(echo ${start} | cut -d"." -f2)
    en=$(echo ${end} | cut -d"." -f1)_$(echo ${end} | cut -d"." -f2)

    sox ${inp_wav_dir}/${ses}_${channel}* ${out_dir}/${st}-${en}@${channel}.wav trim $start $dur

  done < ${seg_file}
done < ${ses_file}

