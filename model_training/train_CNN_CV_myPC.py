import os
import glob
import numpy as np
import pickle
from keras.optimizers import Adam, RMSprop, Adagrad
from model_CNN import create_network_baseline, train_noDataGen
from DataGeneratorClass_CNN import DataGenerator
from data_processing import feat_splice, get_data_shuffle_CV
import pdb


def main(feat_dir, model_dir):

    train_feat_dir = "{0}/train/".format(feat_dir)
    val_feat_dir = "{0}/val/".format(feat_dir)

    # Data gneerator parameters
    data_gen_params = {'batch_size': 100
                       }

    # Training params Baseline (DNN)

    # As input frame has 13 mfcc features and 1 pitch feature. +/-3 frames splicing. delta, delta-deltas.. 14*7*3=294
    # 32 spliced frames per batch

    feat_dim = 13
    segment_length = 100
    batch_size = data_gen_params['batch_size']
    input_shape = (feat_dim, segment_length, 1)

    training_params_DNN = {'input_shape': input_shape,
                           'num_epochs': 2,
                           'num_folds_CV': 5,
                           'batch_size': 32,
                           'num_FC_layers': 2,
                           'num_conv_layers': 2,
                           'kernels': (64, 128),
                           'kernel_size': ((3, 3), (2, 2)),
                           'num_FC_units': (512, 256),
                           'dropout_rate': 0.2,
                           'Batch_norm_FLAG': True,
                           'Batch_norm_momentum': 0.99,
                           'l1_regularizer_weight': 0,
                           'init_learning_rate': 0.001
                           }

    # Initialize model
    model = create_network_baseline(training_params_DNN)
    model.summary()
    # Compile model
    model.compile(optimizer=RMSprop(lr=training_params_DNN['init_learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Logging directory
    log_dir = model_dir + 'log_CV/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create train and val data
    all_feats = np.array([[0, 0], [1, 1]])
    all_labels = np.array([0, 0, 0])

    data_success_FLAG = True
    while (all_feats.shape[0] != all_labels.shape[0]) or (not data_success_FLAG):
        all_feats, all_labels, all_num_segments = get_data_shuffle_CV(train_feat_dir, val_feat_dir)
        if all_feats.shape[0] == 2:
            data_success_FLAG = False
        else:
            data_success_FLAG = True

    num_folds = training_params_DNN['num_folds_CV']
    num_segments_val = np.floor(int(all_num_segments) / int(num_folds))

    for fold in range(num_folds):
        print("fold = " + str(fold))
        val_indices = list(range(int(fold*num_segments_val), int((fold+1)*num_segments_val)))
        train_indices = [x for x in list(range(all_num_segments)) if x not in val_indices]
        log_dir_fold = log_dir + '/fold_{0}'.format(fold)

        if not os.path.exists(log_dir_fold):
            os.makedirs(log_dir_fold)
        train_noDataGen(model, (all_feats[train_indices], all_labels[train_indices]), (all_feats[val_indices], all_labels[val_indices]),
                        log_dir_fold,
                        epochs=training_params_DNN['num_epochs'],
                        batch_size=training_params_DNN['batch_size'])
                            #train_steps_per_epoch=np.floor(train_num_segments/training_params_DNN['batch_size']))

if __name__ == '__main__':

    feat_inp_dir = '/media/External_HD/tiles_audio/train_test_CNN/'

    speech_features_dir = '{0}/'.format(feat_inp_dir)

    model_dir = '/media/External_HD/tiles_audio/models/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    main(speech_features_dir, model_dir)
