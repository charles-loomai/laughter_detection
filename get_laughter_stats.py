
from bs4 import BeautifulSoup as bs
import argparse

""" This script takes as input an ICSI transcript file (.mrt) and returns the duration statistics of the 4 different kinds of laughters:
    1) laugh alone segments
    2) laughing during uttering a word
    3) laughing in utterance but not during a word
    4) breath laugh

    python get_laughter_stats.py <input_transcript>

    Returns a tuple with the above durations in same order
"""


def get_laughter_stats(inp_transcript, out_dir):
    with open(inp_transcript,'r') as f:
        contents = f.read()

    #print(contents)
    xml_file = bs(contents)

    while_laughing = [wl for wl in xml_file.find_all('comment') if wl['description'] == 'while laughing']
    breath_laugh = [wl for wl in xml_file.find_all('vocalsound') if wl['description'] == 'breath-laugh']
    laugh = [wl for wl in xml_file.find_all('vocalsound') if 'laugh' in wl['description']]

    bl_file = open("{0}/breath_laugh_segments.txt".format(out_dir), 'w')
    for seg in [bl for bl in xml_file.find_all('segment') if bl.find('vocalsound')]:
        if not seg.find('comment') or seg.find('comment')['description'] != 'Digits':
            if 'breath-laugh' in seg.find('vocalsound')['description']:
                bl_file.write("{0} {1}\n".format(seg['starttime'], seg['endtime']))

    bl_file.close()

    l_file = open("{0}/laugh_segments.txt".format(out_dir), 'w')
    for seg in [bl for bl in xml_file.find_all('segment') if bl.find('vocalsound')]:
        if not seg.find('comment') or seg.find('comment')['description'] != 'Digits':
            if 'laugh' in seg.find('vocalsound')['description'] and 'breath-laugh' not in seg.find('vocalsound')['description']:
                l_file.write("{0} {1}\n".format(seg['starttime'], seg['endtime']))

    l_file.close()

    wl_file = open("{0}/while_laughing_segments.txt".format(out_dir), 'w')
    for seg in [bl for bl in xml_file.find_all('segment') if bl.find('comment')]:
        if 'while laughing' in seg.find('comment')['description']:
            wl_file.write("{0} {1}\n".format(seg['starttime'], seg['endtime']))
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