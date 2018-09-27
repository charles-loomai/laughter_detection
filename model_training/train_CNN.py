import os
import glob
import numpy as np
import pickle
from keras.optimizers import Adam, RMSprop, Adagrad
from model_CNN import create_network_baseline, train_noDataGen
from DataGeneratorClass_CNN import DataGenerator
from data_processing import feat_splice, get_data_shuffle
import pdb
import sys
import argparse
from keras.callbacks import Callback

def main(feat_dir, model_dir, iteration):

    train_feat_dir = "{0}/train/".format(feat_dir)
    val_feat_dir = "{0}/val/".format(feat_dir)
    # Data gneerator parameters

    # Training params Baseline (DNN)

    # As input frame has 13 mfcc features and 1 pitch feature. +/-3 frames splicing. delta, delta-deltas.. 14*7*3=294
    # 32 spliced frames per batch

    feat_dim = 13
    segment_length = 100
    input_shape = (feat_dim, segment_length, 1)

    training_params_DNN = {'input_shape': input_shape,
                           'num_epochs': 30,
                           'batch_size': 8,
                           'num_FC_layers': 2,
                           'num_conv_layers': 2,
                           'kernels': (128, 256),
                           'kernel_size': ((3, 3), (2, 2)),
                           'num_FC_units': (256, 128), #(128, 32),
                           'dropout_rate': 0.4,
                           'Batch_norm_FLAG': True,
                           'Batch_norm_momentum': 0.99,
                           'l1_regularizer_weight': 0,
                           'init_learning_rate': 0.0001
                           }

    # Initialize model
    model = create_network_baseline(training_params_DNN)
    model.summary()
    # Compile model
    model.compile(optimizer=RMSprop(lr=training_params_DNN['init_learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    # Logging directory
    log_dir = model_dir + 'log_{0}/'.format(iteration)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create train and val data
    train_feats = np.array([[0, 0], [1, 1]])
    train_labels = np.array([0, 0, 0])
    val_feats = np.array([[0, 0], [1, 1]])
    val_labels = np.array([0, 0, 0])

    data_success_FLAG = True

    while (train_feats.shape[0] != train_labels.shape[0]) or (not data_success_FLAG):
        train_feats, train_labels, train_num_segments = get_data_shuffle(train_feat_dir)
        if train_feats.shape[0] == 2:
            data_success_FLAG = False
        else:
            data_success_FLAG = True
    
    while (val_feats.shape[0] != val_labels.shape[0]) or (not data_success_FLAG):
        val_feats, val_labels, val_num_segments = get_data_shuffle(val_feat_dir)
        if val_feats.shape[0] == 2:
            data_success_FLAG = False
        else:
            data_success_FLAG = True
    print(train_num_segments)
    print(val_num_segments)
    print("Succesfully extracted samples")
    train_noDataGen(model, (train_feats, train_labels), (val_feats, val_labels),
                            log_dir, 
                            epochs=training_params_DNN['num_epochs'], 
                            batch_size=training_params_DNN['batch_size'])


if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument('iter',
                    help='the iteration number for evaluating multiple times same model')

    args = parser.parse_args()

    feat_inp_dir = '/work/rperi/laugh/data/train_test_CNN/'

    speech_features_dir = '{0}/'.format(feat_inp_dir)

    model_dir = '/work/rperi/laugh/data/models_CNN/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    main(speech_features_dir, model_dir, args.iter)
