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


def extract_mfcc_from_uniform(segments_file, features_file):

    with open(segments_file, 'r') as f:
        segments = f.readlines()[1:]

    with open(features_file, 'r') as f:
        features = f.readlines()

    feats_mfcc = []
    feats_mfcc_de = []
    feats_mfcc_de_de = []

    mfcc_all_indices = [x for x, elem in enumerate(features[0].split(';')) if 'mfcc_sma' in elem]

    mfcc_indices = mfcc_all_indices[0:14]
    mfcc_de_indices = mfcc_all_indices[14:28]
    mfcc_de_de_indices = mfcc_all_indices[28:42]

    start_times_segments = [float(seg.split(',')[0]) for seg in segments]
    start_times_features = [float(feat.split(';')[1]) for feat in features[1:]]

    for seg_times in start_times_segments:
        seg = round(seg_times, 2)  # Rounds to 2 decimal points. This is the precision offered in the features
        if seg in start_times_features:
            line_num = int(start_times_features.index(seg))
        else:
            print("Time stamp from segments not found in features")
            print(segments_file, seg_times)
            exit(1)

        feats_mfcc.append([float(f) for idx, f in enumerate(features[line_num+1].split(';')) if idx in mfcc_indices])
        feats_mfcc_de.append([float(f) for idx, f in enumerate(features[line_num + 1].split(';')) if idx in mfcc_de_indices])
        feats_mfcc_de_de.append([float(f) for idx, f in enumerate(features[line_num + 1].split(';')) if idx in mfcc_de_de_indices])

    feats_mfcc = np.transpose(np.array(feats_mfcc))
    feats_mfcc_de = np.transpose(np.array(feats_mfcc_de))
    feats_mfcc_de_de = np.transpose(np.array(feats_mfcc_de_de))

    return feats_mfcc, feats_mfcc_de, feats_mfcc_de_de


if __name__=='__main__':

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

        for seg_file in segment_files:
            chan = seg_file.split('/')[-1].split('_')[1]
            features_files = glob.glob(os.path.join(inp_feats_dir, '{0}_{1}*'.format(session_id, chan)))[0]

            (mfcc, mfcc_de, mfcc_de_de) = extract_mfcc_from_uniform(seg_file, features_files)

            with open('{0}/{1}_mfcc.pickle'.format(out_feat_dir, chan), 'wb') as f:
                pickle.dump(mfcc, f)

            with open('{0}/{1}_mfcc-de.pickle'.format(out_feat_dir, chan), 'wb') as f:
                pickle.dump(mfcc_de, f)

            with open('{0}/{1}_mfcc-de-de.pickle'.format(out_feat_dir, chan),'wb') as f:
                pickle.dump(mfcc_de_de, f)
