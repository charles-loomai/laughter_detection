import os
import glob
import numpy as np
import pickle
from keras.optimizers import Adam, RMSprop, Adagrad
from model import create_network_baseline, train, train_noDataGen
from DataGeneratorClass import DataGenerator
from data_processing import feat_splice, get_data_shuffle_DNN
import pdb
from keras.callbacks import Callback

def main(feat_dir, model_dir):

    train_feat_dir = "{0}/train/".format(feat_dir)
    val_feat_dir = "{0}/val/".format(feat_dir)

    # Data gneerator parameters
    data_gen_params = {'splice_context': 25,
                       'inp_dim': (294,),
                       'target_dim': (1, 1),
                       'batch_size': 100
                       }

    # Training params Baseline (DNN)

    # As input frame has 13 mfcc features and 1 pitch feature. +/-3 frames splicing. delta, delta-deltas.. 14*7*3=294
    # 32 spliced frames per batch
    feat_dim = 13

    training_params_DNN = {'num_epochs': 10,
                           'batch_size': 64,
                           'splice_context': 25,
                           'num_FC_layers': 4,
                           'num_FC_units': (256, 128, 64, 32),
                           'dropout_rate': 0.2,
                           'Batch_norm_FLAG': True,
                           'Batch_norm_momentum': 0.99,
                           'l1_regularizer_weight': 0,
                           'init_learning_rate': 0.001
                           }

    spliced_dim = feat_dim * (2 * training_params_DNN['splice_context'] + 1)
    print(spliced_dim)
    training_params_DNN['input_shape'] = (spliced_dim,)

    # Initialize model
    model = create_network_baseline(training_params_DNN)
    model.summary()
    # Compile model
    model.compile(optimizer=RMSprop(lr=training_params_DNN['init_learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Create train and val data
    train_feats = np.array([[0,0],[1,1]])
    train_labels = np.array([0, 0, 0])
    val_feats = np.array([[0, 0], [1, 1]])
    val_labels = np.array([0, 0, 0])

    while (train_feats.shape[0] != train_labels.shape[0]):
        train_feats, train_labels = get_data_shuffle_DNN(train_feat_dir, training_params_DNN['splice_context'])
    while (val_feats.shape[0] != val_labels.shape[0]):
        val_feats, val_labels = get_data_shuffle_DNN(val_feat_dir, training_params_DNN['splice_context'])
    print("Succesfully extracted samples")
    train_noDataGen(model, (train_feats, train_labels), (val_feats, val_labels),
                    epochs=training_params_DNN['num_epochs'],
                    batch_size=training_params_DNN['batch_size'])
    #scores = model.evaluate(val_feats, val_labels, batch_size=training_params_DNN['batch_size'])
    #print(scores)

if __name__ == '__main__':

    feat_inp_dir = '/work/rperi/laugh/data/train_test_CNN/'

    speech_features_dir = '{0}/'.format(feat_inp_dir)

    model_dir = '/work/rperi/laugh/data/models/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    main(speech_features_dir, model_dir)
