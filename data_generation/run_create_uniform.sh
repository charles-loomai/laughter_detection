#!/bin/bash

type_of_sound=laugh  # laugh or speech

segment_length=1  # in seconds


#input_seg_dir=/home/raghuveer/work/TILES/laughter/data/data_generation/${type_of_sound}_segments_chan/ 
input_seg_dir=/home/raghuveer/work/TILES/laughter/data/data_generation/corrected_annotations/

output_seg_dir_base=/home/raghuveer/work/TILES/laughter/data/data_generation/${type_of_sound}_uniform_segments

audio_directory=/media/External_HD/tiles_audio/icsi_close_mic_wav/
#ses=Bdb001
while read -r ses; do
  echo $ses
  out_seg_dir=${output_seg_dir_base}/${ses}

  mkdir -p ${out_seg_dir}
  
  find ${input_seg_dir}/${ses} -name "*.csv" > segments_files
  while read -r file; do
      file_name=$(basename $file .csv)
      channel=$(echo $file_name | awk -F_ '{print $(NF-1)}')
      #echo $channel
      audio_file=${audio_directory}/${ses}_${channel}
      input_audio_duration=$(soxi -D ${audio_file}*)

      out_seg_file=${out_seg_dir}/${file_name}_uniform.csv

      python3 ./create_uniform_laughter_segments.py ${input_audio_duration} $file $out_seg_file $segment_length
  done < segments_files
done < ../../../data/ses_list
