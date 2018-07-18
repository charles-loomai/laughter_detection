#!/bin/bash

segment_length=1
ses=Bdb001
audio_directory=/home/raghuveer/work/TILES/laughter/data/data_generation/audio/${ses}
out_seg_dir=/home/raghuveer/work/TILES/laughter/data/data_generation/uniform_segments/${ses}

mkdir -p ${out_seg_dir}

while read -r seg_file ; do
  file_name=$(basename  $(echo $seg_file | awk -F\/ '{print $NF}') ".csv")
  channel=$(echo $file_name | awk -F_ '{print $NF}')
  #echo $channel
  
  audio_file=${audio_directory}/${ses}_${channel}
  input_audio_duration=$(soxi -D ${audio_file}*)
  #echo $input_audio_duration

  out_seg_file=${out_seg_dir}/${file_name}_uniform.csv

  python3 ./create_uniform_laughter_segments.py ${input_audio_duration} $seg_file $out_seg_file $segment_length
done < ${1}

