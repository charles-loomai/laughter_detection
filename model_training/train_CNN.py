import os
import glob
import numpy as np
import pickle
from keras.optimizers import Adam, RMSprop, Adagrad
from model_CNN import create_network_baseline, train
from DataGeneratorClass_CNN import DataGenerator
from data_processing import feat_splice
import pdb


def main(feat_dir, model_dir):

    train_feat_dir = "{0}/train/".format(feat_dir)
    val_feat_dir = "{0}/val/".format(feat_dir)

    # Data gneerator parameters
    data_gen_params = {'batch_size': 32
                       }

    # Training params Baseline (DNN)

    # As input frame has 13 mfcc features and 1 pitch feature. +/-3 frames splicing. delta, delta-deltas.. 14*7*3=294
    # 32 spliced frames per batch

    feat_dim = 13
    segment_length = 100
    batch_size = data_gen_params['batch_size']
    input_shape = (feat_dim, segment_length, 1)

    training_params_DNN = {'input_shape': input_shape,
                           'num_epochs': 30,
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

    Generator = DataGenerator(**data_gen_params)

    training_gen = Generator.load_DataGenerators(train_feat_dir)
    val_gen = Generator.load_DataGenerators(val_feat_dir)

    model = train(model, training_gen,
                  val_gen,
                  epochs=training_params_DNN['num_epochs'],
                  train_steps_per_epoch=int(np.floor((6000)/data_gen_params['batch_size'])),
                  val_steps_per_epoch=int(np.floor((6000)/data_gen_params['batch_size']))
                  )


if __name__ == '__main__':

    feat_inp_dir = '/media/External_HD/tiles_audio/train_test_CNN/'

    speech_features_dir = '{0}/'.format(feat_inp_dir)

    model_dir = '/media/External_HD/tiles_audio/models/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    main(speech_features_dir, model_dir)
