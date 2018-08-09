import glob
import os
import numpy as np
import pickle
from collections import defaultdict


"""
This script takes in a segments file (with uniform segments) and a lld features file (extracted using opensmile)
and extracts mfcc features from the features file corresponding to each segment

Input args:
  segment_file
  feature_file
Returns
  np array of mfcc features for each segment

"""


def extract_mfcc_from_segments(segments_file, features_file, frame_shift):

    with open(segments_file, 'r') as f:
        segments = f.readlines()

    with open(features_file, 'r') as f:
        features = f.readlines()

    mfcc_all_indices = [x for x, elem in enumerate(features[0].split(';')) if 'mfcc_sma' in elem]
    pitch_indices = [x for x, elem in enumerate(features[0].split(';')) if 'F0_sma' in elem]

    mfcc_indices = mfcc_all_indices[0:13]
    mfcc_de_indices = mfcc_all_indices[14:27]
    mfcc_de_de_indices = mfcc_all_indices[28:41]

    start_times_segments = [float(seg.split(',')[0]) for seg in segments]
    duration_segment = [float(seg.split(',')[2]) for seg in segments]
    start_times_features = [float(feat.split(';')[1]) for feat in features[1:]]

    feats_mfcc = np.empty((1, len(start_times_segments)), dtype=object)
    feats_mfcc_de = np.empty((1, len(start_times_segments)), dtype=object)
    feats_mfcc_de_de = np.empty((1, len(start_times_segments)), dtype=object)

    for idx, seg_times in enumerate(start_times_segments):
        feats_mfcc_seg = []
        feats_mfcc_de_seg = []
        feats_mfcc_de_de_seg = []

        duration = duration_segment[idx]
        num_frames_in_segment = int(np.floor(duration*1000/frame_shift))
        
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

        feats_mfcc[0, idx] = feats_mfcc_seg
        feats_mfcc_de[0, idx] = feats_mfcc_de_seg
        feats_mfcc_de_de[0, idx] = feats_mfcc_de_de_seg

    total_num_frames = 0
    for seg_idx in range(len(segments)):
        total_num_frames += len(feats_mfcc[0, seg_idx])
    return feats_mfcc, feats_mfcc_de, feats_mfcc_de_de, total_num_frames


if __name__=='__main__':

    frame_shift = 10  # in msec
    SpeechOrLaugh = 'speech'  #'speech' or 'laugh'

    with open('../../../data/ses_list', 'r') as f:
        sessions = [ses.strip() for ses in f.readlines()]

    inp_feats_dir = '/media/External_HD/tiles_audio/icsi_close_mic_tiles_lld/'
    out_feat_dir_base = '/media/External_HD/tiles_audio/icsi_feats_mfcc_{0}/'.format(SpeechOrLaugh)

    # Get speaker-session mapping
    spkr_session = defaultdict(list)
    spkrs = []
    for session_id in sessions:
        #inp_seg_dir = '/home/raghuveer/work/TILES/laughter/data/data_generation/corrected_annotations/{0}'.format(session_id)
        inp_seg_dir = '/home/raghuveer/work/TILES/laughter/data/data_generation/{1}_segments_chan/{0}'\
            .format(session_id, SpeechOrLaugh)

        segment_files = glob.glob(os.path.join(inp_seg_dir, '*.csv'))
        # print(segment_files)

        for seg_file in segment_files: #[x for x in segment_files if 'while-laughing' not in x]:
            spkr = seg_file.split('/')[-1].split('_')[2].strip('.csv')
            spkrs.append(spkr)

            if session_id not in spkr_session[spkr]:
                spkr_session[spkr].append(session_id)

    for speaker in spkr_session.keys():
        print(speaker)


        session_ids = spkr_session[speaker]

        for session_id in session_ids:
            #print(session_id)
            num_frames = []

            #inp_seg_dir = '/home/raghuveer/work/TILES/laughter/data/data_generation/corrected_annotations/{0}'.format(session_id)
            inp_seg_dir = '/home/raghuveer/work/TILES/laughter/data/data_generation/{1}_segments_chan/{0}'.\
                format(session_id, SpeechOrLaugh)

            segment_files = glob.glob(os.path.join(inp_seg_dir, '*_{0}.csv'.format(speaker)))
            #print(segment_files)

            all_seg_files = [x for x in segment_files if 'while-laughing' not in x]

            mfcc = np.empty((1,len(all_seg_files)), dtype=object)
            mfcc_de = np.empty((1, len(all_seg_files)), dtype=object)
            mfcc_de_de = np.empty((1, len(all_seg_files)), dtype=object)

            for seg_file_idx, seg_file in enumerate(all_seg_files):
                chan = seg_file.split('/')[-1].split('_')[1]
                spkr = seg_file.split('/')[-1].split('_')[2].strip('.csv')
                features_files = glob.glob(os.path.join(inp_feats_dir, '{0}_{1}*'.format(session_id, chan)))

                if features_files:  # Only if features for corresponding channels are present
                    with open(seg_file, 'r') as f:
                        if f.readlines():
                            out = extract_mfcc_from_segments(seg_file, features_files[0], frame_shift)

                            mfcc[0, seg_file_idx] = out[0]
                            mfcc_de[0, seg_file_idx] = out[1]
                            mfcc_de_de[0, seg_file_idx] = out[2]
                            num_frames.append(out[3])
            #print("Writing out features")

            mfcc_new = np.hstack(list(mfcc[0]))
            mfcc_de_new = np.hstack(list(mfcc_de[0]))
            mfcc_de_de_new = np.hstack(list(mfcc_de_de[0]))

            out_feat_dir = "{0}/{1}/".format(out_feat_dir_base, speaker)

            if not os.path.exists(out_feat_dir):
                os.makedirs(out_feat_dir)

            total_num_frames_spkr = np.sum(num_frames)

            with open('{0}/{2}_{1}_mfcc.pickle'.format(out_feat_dir, total_num_frames_spkr, session_id), 'wb') as f:
                pickle.dump(mfcc_new, f)

            with open('{0}/{2}_{1}_mfcc-de.pickle'.format(out_feat_dir, total_num_frames_spkr, session_id), 'wb') as f:
                pickle.dump(mfcc_de_new, f)

            with open('{0}/{2}_{1}_mfcc-de-de.pickle'.format(out_feat_dir, total_num_frames_spkr, session_id), 'wb') as f:
                pickle.dump(mfcc_de_de_new, f)
