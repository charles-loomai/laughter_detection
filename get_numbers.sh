#!/bin/bash
# This script computes the number of segments per each type of laughter
# Input : 
#       arg1 - directory containing segments
#       arg2 - list of sessions as mrt files

segments_dir=$1
ses_list=$2

laughter_only=0
laughter_within=0
breath=0
while=0

while read -r line; do

    ses=$(basename $line '.mrt')

    num_laughter_only=$(wc -l ${segments_dir}/${ses}/laugh_only_segments.txt)
    num_laughter_within=$(wc -l ${segments_dir}/${ses}/laugh_utterance_segments.txt)
    num_breath=$(wc -l ${segments_dir}/${ses}/breath_laugh_segments.txt)
    num_while=$(wc -l ${segments_dir}/${ses}/while_laughing_segments.txt)

    laughter_only=$(( $laughter_only + $(echo $num_laughter_only | cut -d" " -f1)))
    laughter_within=$(( $laughter_within + $(echo $num_laughter_within | cut -d" " -f1)))
    breath=$(( $breath + $(echo $num_breath | cut -d" " -f1)))
    while=$(( $while + $(echo $num_while | cut -d" " -f1)))

done < ${ses_list}

echo "Number of laughter only segments = "$laughter_only
echo "Number of laughter within utterance segments = "$laughter_within
echo "Number of breath laughter segments = "$breath
echo "Number of laughter within utterance and overlapping with speech segments = "$while
