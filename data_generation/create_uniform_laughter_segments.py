import numpy.random as ran
import numpy as np
import argparse

def make_uniform_from_short(dur, segment_length, start):
    """
    Append more audio to make duration of segment = segment_length
    :return: (start_new, end_new)
    """

    deficit = segment_length - dur

    start_new = 0
    while start_new <= 0:
        if start != 0:
            offset_from_start = ran.uniform(0, -deficit)
            start_new = start + offset_from_start
        else:
            start_new = start
            break

    end_new = start_new + segment_length

    return start_new, end_new


def create_uniform_laughter_segments(inp_audio_duration, segments_file, segment_length):

    """

    This script takes the following inpurt arguments:
    1) inp_audio_duration (in sec)
    2) segments_file
        Format: <start_time(in sec)> <end_time(in sec)>
    3) class_label (laughter, speech or others)
    4) segment_length (in sec)

    """
    with open(segments_file, 'r') as f:
        segments = [seg.strip() for seg in f.readlines()]

    timings_new = []
    for seg in segments:
        st,duration = seg.split(',')[0:3:2]

        start = float(st)
        dur = float(duration)
        end = start + dur

        # Append more audio to the current segment to make total duration = segment_length
        if dur < segment_length:
            start_final, end_final = make_uniform_from_short(dur, segment_length, start)
            if end_final < inp_audio_duration:  # To avoid overshooting the acutal audio
                timings_new.append((start_final, end_final))
            else:
                print("Exceeded audio. Ignoring segment...")

        # Split current segment into multiple uniform segments (of length = segment_length)
        else:
            num_segments = int(np.floor(dur/segment_length))
            for segment_idx in range(num_segments):
                start_new = start + segment_length*segment_idx
                end_new = start + segment_length*(segment_idx+1)
                timings_new.append((start_new, end_new))

            start_final = end_new
            end_final = end_new + segment_length
            if end_final < inp_audio_duration:  # To avoid overshooting the acutal audio
                timings_new.append((start_final, end_final))
            else:
                print("Exceeded audio. Ignoring segment...")


    return timings_new


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Creates uniform segments given a segments file')

    parser.add_argument('inp_aud_dur', type=float, help='Input audio duration. To ensure new segments don\'t overshoot audio')
    parser.add_argument('seg_file', type=str, help='Original segments csv file')
    parser.add_argument('out_seg_file', type=str, help='Output segments file')
    parser.add_argument('segment_length', type=float, help='The new segment length that needs to be created')

    args = parser.parse_args()
    # inp_aud_dur = 3600
    # seg_file = '/home/raghuveer/work/TILES/laughter/data/dummy_segments.csv'
    # class_label = 'laughter'
    # segment_length = 1

    times = create_uniform_laughter_segments(args.inp_aud_dur, args.seg_file, args.segment_length)

    with open(args.out_seg_file, 'w') as o:
        o.write("start_time,duration\n")
        for time in times:
            st, en = time
            dur = en - st
            o.write("{0},{1}\n".format(st, dur))
    x = 0
