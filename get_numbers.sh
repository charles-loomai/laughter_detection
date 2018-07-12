#!/bin/bash

laughter_only=0
laughter_within=0
breath=0
while=0

while read -r line; do

ses=$(basename $line '.mrt')

num_laughter_only=$(wc -l ./laughter_segments/${ses}/laugh_only_segments.txt)
num_laughter_within=$(wc -l ./laughter_segments/${ses}/laugh_utterance_segments.txt)
num_breath=$(wc -l ./laughter_segments/${ses}/breath_laugh_segments.txt)
num_while=$(wc -l ./laughter_segments/${ses}/while_laughing_segments.txt)

laughter_only=$(( $laughter_only + $(echo $num_laughter_only | cut -d" " -f1)))
laughter_within=$(( $laughter_within + $(echo $num_laughter_within | cut -d" " -f1)))
breath=$(( $breath + $(echo $num_breath | cut -d" " -f1)))
while=$(( $while + $(echo $num_while | cut -d" " -f1)))

done < ./transcripts/ses_list

echo $laughter_only
echo $laughter_within
echo $breath
echo $while
