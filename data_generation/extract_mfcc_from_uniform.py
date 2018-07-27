import glob
import os
import numpy as np
import pickle

"""
This script takes in a segments file (with uniform segments) and a lld features file (extracted using opensmile)
and extracts mfcc features from the features file corresponding to each segment

Input args:
  segment_file
  feature_file
Returns
  np array of mfcc features for each segment

"""


def extract_mfcc_from_uniform(segments_file, features_file, num_frames_in_segment):

    with open(segments_file, 'r') as f:
        segments = f.readlines()[1:]

    with open(features_file, 'r') as f:
        features = f.readlines()

    feats_mfcc = []
    feats_mfcc_de = []
    feats_mfcc_de_de = []

    mfcc_all_indices = [x for x, elem in enumerate(features[0].split(';')) if 'mfcc_sma' in elem]
    pitch_indices = [x for x, elem in enumerate(features[0].split(';')) if 'F0_sma' in elem]

    mfcc_indices = mfcc_all_indices[0:13]
    mfcc_de_indices = mfcc_all_indices[14:27]
    mfcc_de_de_indices = mfcc_all_indices[28:41]

    start_times_segments = [float(seg.split(',')[0]) for seg in segments]
    start_times_features = [float(feat.split(';')[1]) for feat in features[1:]]

    for seg_times in start_times_segments:
        feats_mfcc_seg = []
        feats_mfcc_de_seg = []
        feats_mfcc_de_de_seg = []

        seg = round(seg_times, 2)  # Rounds to 2 decimal points. This is the precision offered in the features
        if seg in start_times_features:
            line_num = int(start_times_features.index(seg))
        else:
            print("Time stamp from segments not found in features")
            print(segments_file, seg_times)
            exit(1)

        for feats in features[line_num+1 : line_num+1+num_frames_in_segment]:
            temp = [float(f) for idx, f in enumerate(feats.split(';')) if idx in mfcc_indices]
            temp.append(float(feats.split(';')[pitch_indices[0]]))
            feats_mfcc_seg.append(temp)

            temp = [float(f) for idx, f in enumerate(feats.split(';')) if idx in mfcc_de_indices]
            temp.append(float(feats.split(';')[pitch_indices[1]]))
            feats_mfcc_de_seg.append(temp)

            temp = [float(f) for idx, f in enumerate(feats.split(';')) if idx in mfcc_de_de_indices]
            temp.append(float(feats.split(';')[pitch_indices[2]]))
            feats_mfcc_de_de_seg.append(temp)

        feats_mfcc.append(feats_mfcc_seg)
        feats_mfcc_de.append(feats_mfcc_de_seg)
        feats_mfcc_de_de.append(feats_mfcc_de_de_seg)

    feats_mfcc = np.transpose(np.array(feats_mfcc))
    feats_mfcc_de = np.transpose(np.array(feats_mfcc_de))
    feats_mfcc_de_de = np.transpose(np.array(feats_mfcc_de_de))
    #print(segments)
    return feats_mfcc, feats_mfcc_de, feats_mfcc_de_de, len(segments)


if __name__=='__main__':

    uniform_seg_length = 1  # in seconds
    frame_shift = 10  # in msec

    num_frames_in_segment = int(uniform_seg_length*1000/frame_shift)  # Assuming 10 msec frame shift and a 1sec uniform segment

    with open('../../../data/ses_list', 'r') as f:
        sessions = [ses.strip() for ses in f.readlines()]

    for session_id in sessions:
        print(session_id)
        inp_seg_dir = '/home/raghuveer/work/TILES/laughter/data/data_generation/speech_uniform_segments/{0}'.format(session_id)
        inp_feats_dir = '/media/External_HD/tiles_audio/icsi_close_mic_tiles_lld/'
        out_feat_dir = '/media/External_HD/tiles_audio/icsi_feats_mfcc_speech/{0}'.format(session_id)

        if not os.path.exists(out_feat_dir):
            os.makedirs(out_feat_dir)

        segment_files = glob.glob(os.path.join(inp_seg_dir, '*.csv'))
        #print(segment_files)

        for seg_file in segment_files:
            chan = seg_file.split('/')[-1].split('_')[1]
            spkr = seg_file.split('/')[-1].split('_')[2]
            features_files = glob.glob(os.path.join(inp_feats_dir, '{0}_{1}*'.format(session_id, chan)))
            if features_files:  # Only if features for corresponding channels are present
                (mfcc, mfcc_de, mfcc_de_de, num_seg) = extract_mfcc_from_uniform(seg_file, features_files[0], num_frames_in_segment)

                with open('{0}/{3}_{1}_{2}_{4}_mfcc.pickle'.format(out_feat_dir, chan, spkr, session_id, num_seg), 'wb') as f:
                    pickle.dump(mfcc, f)

                with open('{0}/{3}_{1}_{2}_{4}_mfcc-de.pickle'.format(out_feat_dir, chan, spkr, session_id, num_seg), 'wb') as f:
                    pickle.dump(mfcc_de, f)

                with open('{0}/{3}_{1}_{2}_{4}_mfcc-de-de.pickle'.format(out_feat_dir, chan, spkr, session_id, num_seg),'wb') as f:
                    pickle.dump(mfcc_de_de, f)
