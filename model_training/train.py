import os
import glob
from random import shuffle


def main(feat_dir, model_dir):

    feats_train_dir = "{0}/train/".format(feat_dir)
    feats_test_dir = "{0}/test".format(feat_dir)

    with open(feats_train_dir+'/train_spkr.list') as f:
        train_spkrs = [x.strip() for x in f.readlines()]

    mfcc_feat_files = []
    mfcc_de_feat_files = []
    mfcc_de_de_feat_files = []
    for spkr in train_spkrs:
        mfcc_feat_files.append(glob.glob(os.path.join('{0}/{1}/'.format(feats_train_dir, spkr), "*_mfcc.pickle")))
        mfcc_de_feat_files.append(glob.glob(os.path.join('{0}/{1}/'.format(feats_train_dir, spkr), "*_mfcc_de.pickle")))
        mfcc_de_de_feat_files.append(glob.glob(os.path.join('{0}/{1}/'.format(feats_train_dir, spkr), "*_mfcc_de_de.pickle")))


    shuffle(train_feat_files)
    x = 0


if __name__=='__main__':

    feat_inp_dir = '/media/External_HD/tiles_audio/train_test/'

    speech_features_dir = '{0}/speech/'.format(feat_inp_dir)

    model_dir = '/media/External_HD/tiles_audio/models/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    main(speech_features_dir, model_dir)