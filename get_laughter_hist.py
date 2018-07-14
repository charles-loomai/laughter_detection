import matplotlib.pyplot as plt
import numpy as np
"""
This script


"""

def get_laughter_hist(segments_dir, ses_list):

    with open(ses_list,'r') as f:
        sessions = f.readlines()

    laugh_only_total = 0
    duration = []
    for ses in sessions:
        if 'Bmr' in ses.strip():
            # Laugh only segments
            with open("{0}/{1}/laugh_utterance_segments.txt".format(segments_dir, ses.strip()), 'r') as f:
                lines = f.readlines()

            for seg in lines:
                start, end, chan = seg.strip().split()
                duration.append(float(end) - float(start))
                laugh_only_total += duration[-1]
    #plt.hist(duration, [0,0.5,1,1.5,2,2.5,3,3.5,4,4.5,5,5.5,6,6.5,7,7.5,8,8.5,9,9.5,10])
    #plt.show()
    print(np.mean(duration))
    print(np.std(duration))
    x = 0


if __name__ == "__main__":
    seg_dir = '/home/raghuveer/work/TILES/work_in_progress/data/laughter_segments/'
    ses_list = '/home/raghuveer/work/TILES/work_in_progress/data/ses_list'

    get_laughter_hist(seg_dir, ses_list)