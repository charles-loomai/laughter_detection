
from bs4 import BeautifulSoup as bs
import argparse
from collections import defaultdict

""" This script takes as input an ICSI transcript file (.mrt) and returns the duration statistics of the 4 different kinds of laughters:
    1) laugh alone segments
    2) laugh during utterance but NOT overlapping with speech
    3) laughing during utterance overlapping with speech (while lauging)
    4) breath laugh

    python create_laughter_segments.py <input_transcript> <output_directory>

    Returns segments information of the different laughter types in output directory 
"""


def create_speech_segments(inp_transcript, out_dir):
    with open(inp_transcript,'r') as f:
        contents = f.read()

    xml_file = bs(contents)

    # Get channel information for each speaker
    channel_dict = defaultdict()
    for speaker in [spk for spk in xml_file.find_all('participant') if spk.has_attr('channel')]:
        channel_dict[speaker['name']] = speaker['channel']

    # Write out timing information for breath-laugh segments
    for seg in [bl for bl in xml_file.find_all('segment') if not (bl.find('vocalsound') or bl.find('nonvocalsound')) and bl.has_attr('participant')]:
        spkr = seg['participant']
        if spkr in channel_dict.keys():
            chan = channel_dict[spkr]
            if not seg.find('comment') or seg.find('comment')['description'] != 'Digits':
                start = float(seg['starttime'])
                duration = float(seg['endtime']) - start
                speechseg_file = open("{0}/speech-segments_{1}_{2}.csv".format(out_dir, chan, spkr), 'a')
                speechseg_file.write("{0},dummy,{1}\n".format(start, duration))

                speechseg_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract laughter segments per type of laughter')
    parser.add_argument('inp_transcript', type=str, help="Input ICSI transcript file in mrt format")
    parser.add_argument('out_dir', type=str, help="Output directory for segments files")

    args = parser.parse_args()

    inp_transcript = args.inp_transcript
    out_dir = args.out_dir
    create_speech_segments(inp_transcript, out_dir)
