
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


def create_laughter_segments(inp_transcript, out_dir):
    with open(inp_transcript,'r') as f:
        contents = f.read()

    xml_file = bs(contents)

    # Get channel information for each speaker
    channel_dict = defaultdict()
    for speaker in [spk for spk in xml_file.find_all('participant') if spk.has_attr('channel')]:
        channel_dict[speaker['name']] = speaker['channel']

    # Write out timing information for breath-laugh segments
    bl_file = open("{0}/breath_laugh_segments.txt".format(out_dir), 'w')
    for seg in [bl for bl in xml_file.find_all('segment') if bl.find('vocalsound') and bl.has_attr('participant')]:
        if seg['participant'] in channel_dict.keys():
            if not seg.find('comment') or seg.find('comment')['description'] != 'Digits':
                vocal_descriptions = [voc['description'] for voc in seg.find_all('vocalsound')]
                for voc in vocal_descriptions:
                    if 'breath-laugh' in voc:
                        bl_file.write("{0} {1} {2}\n".format(seg['starttime'], seg['endtime'], channel_dict[seg['participant']]))

    bl_file.close()

    # Write out timing information for laugh only segments without any speech
    l_only_file = open("{0}/laugh_only_segments.txt".format(out_dir), 'w')

    # Write out timing information for laugh segments within utterance but not overlapping with word
    l_file = open("{0}/laugh_utterance_segments.txt".format(out_dir), 'w')

    for seg in [bl for bl in xml_file.find_all('segment') if bl.find('vocalsound') and bl.has_attr('participant')]:
        if seg['participant'] in channel_dict.keys():
            if not seg.find('comment') or seg.find('comment')['description'] != 'Digits':
                vocal_descriptions = [voc['description'] for voc in seg.find_all('vocalsound')]
                for voc in vocal_descriptions:
                    if 'laugh' in voc and 'breath-laugh' not in voc:
                        if not "{0}".format(seg.contents[0]).strip():  # The segment doesn't contain any speech. Only laughter
                            l_only_file.write("{0} {1} {2}\n".format(seg['starttime'], seg['endtime'], channel_dict[seg['participant']]))
                        else:
                            l_file.write("{0} {1} {2}\n".format(seg['starttime'], seg['endtime'], channel_dict[seg['participant']]))

    l_file.close()
    l_only_file.close()

    # Write out timing information for while-laughing segments
    wl_file = open("{0}/while_laughing_segments.txt".format(out_dir), 'w')
    for seg in [bl for bl in xml_file.find_all('segment') if bl.find('comment') and bl.has_attr('participant')]:
        if seg['participant'] in channel_dict.keys():
            com_descriptions = [com['description'] for com in seg.find_all('comment')]
            for desc in com_descriptions:
                if 'while laughing' in desc:
                    wl_file.write("{0} {1} {2}\n".format(seg['starttime'], seg['endtime'], channel_dict[seg['participant']]))
    wl_file.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Extract laughter segments per type of laughter')
    parser.add_argument('inp_transcript', type=str, help="Input ICSI transcript file in mrt format")
    parser.add_argument('out_dir', type=str, help="Output directory for segments files")

    args = parser.parse_args()

    inp_transcript = args.inp_transcript
    out_dir = args.out_dir
    create_laughter_segments(inp_transcript, out_dir)
