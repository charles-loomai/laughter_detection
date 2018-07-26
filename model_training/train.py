import os
import glob
from random import shuffle
import pickle
import numpy as np


def padding(feat_array, splice_context):
    feat_dimension, num_frames = feat_array.shape
    feat_array_padded = np.zeros((feat_dimension, num_frames + 2*splice_context))
    feat_array_padded[:, splice_context:num_frames + splice_context] = feat_array[:,:]
    for ctx in range(splice_context):
        feat_array_padded[:, ctx] = feat_array[:,0]
        feat_array_padded[:, num_frames + ctx] = feat_array[:, -1]

    return feat_array_padded


def feat_splice(feature_array, splice_context):

    feat_dim, num_frames, num_segments = feature_array.shape
    feat_array_spliced = np.zeros((feat_dim*(splice_context*2 + 1), num_frames, num_segments))
    for segments in range(num_segments):
        feat_array_padded = padding(feature_array[:, :, segments], splice_context)

        for frame in range(num_frames):
            for splice in range(splice_context*2 + 1):
                feat_array_spliced[splice*feat_dim: feat_dim + splice*feat_dim, frame, segments] = feat_array_padded[:, frame + splice]

    return feat_array_spliced


def load_feats(feats_dir, trainORtest, splice_context, required_num_segments):
    feats_train_dir = "{0}/{1}/".format(feats_dir, trainORtest)

    with open(feats_train_dir + '/{0}_spkr.list'.format(trainORtest)) as f:
        train_spkrs = [x.strip() for x in f.readlines()]

    mfcc_feat_files = []
    mfcc_de_feat_files = []
    mfcc_de_de_feat_files = []

    for spkr in train_spkrs:
        mfcc_feat_files.append(glob.glob(os.path.join('{0}/speech/{1}/'.format(feats_train_dir, spkr), "*_mfcc.pickle")))
    mfcc_feat_files = [y for x in mfcc_feat_files for y in x]
    shuffle(mfcc_feat_files)

    # Get corresponding delta, delta-delta features
    for feats in mfcc_feat_files:
        mfcc_de_feat_files.append("{0}/{1}_mfcc-de.pickle".format("/".join(feats.split("/")[0:-1]),
                                                                  "_".join(feats.split("/")[-1].split("_")[0:2])))
        mfcc_de_de_feat_files.append("{0}/{1}_mfcc-de-de.pickle".format("/".join(feats.split("/")[0:-1]),
                                                                        "_".join(feats.split("/")[-1].split("_")[0:2])))



    # Extract one file to get feature dimension and frames
    feat = pickle.load(open(mfcc_feat_files[0], 'rb'))
    feat_dim, num_frames, segments = feat.shape

    feats_spliced = np.zeros((feat_dim*(2*splice_context + 1), num_frames, required_num_segments))
    feats_de_spliced = np.zeros((feat_dim * (2 * splice_context + 1), num_frames, required_num_segments))
    feats_de_de_spliced = np.zeros((feat_dim * (2 * splice_context + 1), num_frames, required_num_segments))

    total_segments = 0
    while total_segments <= required_num_segments:
        for idx, files in enumerate(mfcc_feat_files):
            # mfcc
            feat = pickle.load(open(files, 'rb'))
            total_segments += feat.shape[2]
            if total_segments > required_num_segments:
                segments_required = required_num_segments - (total_segments - feat.shape[2])
                feats_spliced[:, :, total_segments - feat.shape[2]: total_segments] = feat_splice(feat[:, :, 0:segments_required], splice_context)
                break
            else:
                feats_spliced[:, :, total_segments - feat.shape[2] : total_segments] = feat_splice(feat, splice_context)

            # delta
            feat = pickle.load(open(mfcc_de_feat_files[idx], 'rb'))
            if total_segments > required_num_segments:
                segments_required = required_num_segments - (total_segments - feat.shape[2])
                feats_de_spliced[:, :, total_segments - feat.shape[2]: total_segments] = \
                    feat_splice(feat[:, :, 0:segments_required], splice_context)
                break
            else:
                feats_de_spliced[:, :, total_segments - feat.shape[2]: total_segments] = \
                    feat_splice(feat, splice_context)

            # delta-delta
            feat = pickle.load(open(mfcc_de_de_feat_files[idx], 'rb'))
            if total_segments > required_num_segments:
                segments_required = required_num_segments - (total_segments - feat.shape[2])
                feats_de_de_spliced[:, :, total_segments - feat.shape[2]: total_segments] = feat_splice(
                    feat[:, :, 0:segments_required], splice_context)
                break
            else:
                feats_de_de_spliced[:, :, total_segments - feat.shape[2]: total_segments] = feat_splice(feat,
                                                                                                     splice_context)

    return feats_spliced, feats_de_spliced, feats_de_de_spliced


def main(feat_dir, model_dir):

    # Data parameters
    splice_context = 3
    num_laugh_segments = 5000  # minority class number of samples
    # Training params Baseline (DNN)

    # As input frame has 13 mfcc features. +/-3 frames splicing. delta, delta-deltas.. 273 = 13*7*3
    # 100 frames per segment (uniform 1 sec segments)
    training_params_DNN = {'input_shape': (273, 100),
                           'num_epochs': 10,
                           'num_FC_layers': 2,
                           'num_FC_units': (400, 400),
                           'dropout_rate': 0.2,
                           'Batch_norm_FLAG': True,
                           'Batch_norm_momentum': 0.99
                           }

    # Splice and Load pre-extracted train/test features
    feats_train, feats_de_train, feats_de_de_train = load_feats(feat_dir, 'train', splice_context, num_laugh_segments)
    feats_test, feats_de_test, feats_de_de_test = load_feats(feat_dir, 'test', splice_context, num_laugh_segments)

    x = 0


if __name__ == '__main__':

    feat_inp_dir = '/media/External_HD/tiles_audio/train_test/'

    speech_features_dir = '{0}/'.format(feat_inp_dir)

    model_dir = '/media/External_HD/tiles_audio/models/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    main(speech_features_dir, model_dir)