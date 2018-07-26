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
    feat_array_spliced = np.zeros((feat_dim*splice_context, num_frames, num_segments))
    for segments in range(num_segments):
        feat_array_padded = padding(feature_array[:,:,segments], splice_context)
        x = 0
        #for frames in range(num_frames):
         #   feat_array_spliced[:,frames,segments] =


def load_feats(feats_dir, trainORtest):
    feats_train_dir = "{0}/{1}/".format(feats_dir, trainORtest)

    with open(feats_train_dir + '/{0}_spkr.list'.format(trainORtest)) as f:
        train_spkrs = [x.strip() for x in f.readlines()]

    mfcc_feat_files = []
    mfcc_de_feat_files = []
    mfcc_de_de_feat_files = []

    for spkr in ['fe016']:
        mfcc_feat_files.append(glob.glob(os.path.join('{0}/speech/{1}/'.format(feats_train_dir, spkr), "*_mfcc.pickle")))
    mfcc_feat_files = [y for x in mfcc_feat_files for y in x]
    shuffle(mfcc_feat_files)

    # Get corresponding delta, delta-delta features
    for feats in mfcc_feat_files:
        mfcc_de_feat_files.append("{0}/{1}_mfcc-de.pickle".format("/".join(feats.split("/")[0:-1]),
                                                                  "_".join(feats.split("/")[-1].split("_")[0:2])))
        mfcc_de_de_feat_files.append("{0}/{1}_mfcc-de-de.pickle".format("/".join(feats.split("/")[0:-1]),
                                                                        "_".join(feats.split("/")[-1].split("_")[0:2])))
    for files in mfcc_feat_files:
        feat = pickle.load(open(files, 'rb'))
        feat = np.transpose(feat)
        feat_splice(feat, 3)
        x = 0
    return mfcc_feat_files, mfcc_de_feat_files , mfcc_de_de_feat_files


def main(feat_dir, model_dir):

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

    # Load pre-extracted train/test features
    mfcc_train, mfcc_de_train, mfcc_de_de_train = load_feats(feat_dir, 'train')
    mfcc_test, mfcc_de_test, mfcc_de_de_test = load_feats(feat_dir, 'test')

    x = 0


if __name__ == '__main__':

    feat_inp_dir = '/media/External_HD/tiles_audio/train_test/'

    speech_features_dir = '{0}/'.format(feat_inp_dir)

    model_dir = '/media/External_HD/tiles_audio/models/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    main(speech_features_dir, model_dir)