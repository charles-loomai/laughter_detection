
from bs4 import BeautifulSoup as bs
import argparse
from collections import defaultdict

""" This script takes as input an ICSI transcript file (.mrt) and returns the duration statistics of the 4 different kinds of laughters:
    1) laugh alone segments
    2) laughing during uttering a word
    4) breath laugh

    python get_laughter_stats.py <input_transcript>

    Returns a directory containing segments information of the different laughters 
"""


def get_laughter_stats(inp_transcript, out_dir):
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

    # Write out timing information for laugh only segments
    l_file = open("{0}/laugh_segments.txt".format(out_dir), 'w')
    for seg in [bl for bl in xml_file.find_all('segment') if bl.find('vocalsound') and bl.has_attr('participant')]:
        if seg['participant'] in channel_dict.keys():
            if not seg.find('comment') or seg.find('comment')['description'] != 'Digits':
                vocal_descriptions = [voc['description'] for voc in seg.find_all('vocalsound')]
                for voc in vocal_descriptions:
                    if 'laugh' in voc and 'breath-laugh' not in voc:
                        l_file.write("{0} {1} {2}\n".format(seg['starttime'], seg['endtime'], channel_dict[seg['participant']]))

    l_file.close()

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

    parser = argparse.ArgumentParser(description='Extract duration statistics of various kinds of laughter')
    parser.add_argument('inp_transcript', type=str, help="Input ICSI transcript file")
    parser.add_argument('out_dir', type=str, help="Output directory for segments files")

    args = parser.parse_args()

    inp_transcript = args.inp_transcript
    out_dir = args.out_dir
    #print(inp_transcript)
    get_laughter_stats(inp_transcript, out_dir)