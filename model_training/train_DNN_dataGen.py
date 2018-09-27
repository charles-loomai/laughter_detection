import os
import glob
import numpy as np
import pickle
from keras.optimizers import Adam, RMSprop, Adagrad
from model import create_network_baseline, train
from DataGeneratorClass import DataGenerator
from data_processing import feat_splice
import pdb

os.environ['KERAS_BACKEND'] = 'tensorflow'

def main(feat_dir, model_dir):

    train_feat_dir = "{0}/train/".format(feat_dir)
    val_feat_dir = "{0}/val/".format(feat_dir)

    # Data gneerator parameters
    data_gen_params = {'splice_context': 25,
                       'inp_dim': (294,),
                       'target_dim': (1, 1),
                       'batch_size': 128
                       }

    # Training params Baseline (DNN)

    # As input frame has 13 mfcc features and 1 pitch feature. +/-3 frames splicing. delta, delta-deltas.. 14*7*3=294
    # 32 spliced frames per batch
    feat_dim = 13
    spliced_dim = feat_dim*(2*data_gen_params['splice_context'] + 1)
    print(spliced_dim)
    training_params_DNN = {'input_shape': (spliced_dim,),
                           'num_epochs': 30,
                           'num_FC_layers': 4,
                           'num_FC_units': (256, 128, 64, 32),
                           'dropout_rate': 0.2,
                           'Batch_norm_FLAG': True,
                           'Batch_norm_momentum': 0.99,
                           'l1_regularizer_weight': 0,
                           'init_learning_rate': 0.0001
                           }

    # Initialize model
    model = create_network_baseline(training_params_DNN)
    model.summary()
    # Compile model
    model.compile(optimizer=Adam(lr=training_params_DNN['init_learning_rate']),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    #model.compile(optimizer=Adagrad(lr=training_params_DNN['init_learning_rate']),
    #              loss='binary_crossentropy',
    #              metrics=['accuracy'])
    # Validation data
    splice_context = data_gen_params['splice_context']

    validation_directory_laugh = '/media/External_HD/tiles_audio/train_test/val/laugh/me018/'
    val_files_laugh = glob.glob(os.path.join(validation_directory_laugh, "*_mfcc.pickle"))

    validation_directory_speech = '/media/External_HD/tiles_audio/train_test/val/speech/me028/'
    val_files_speech = glob.glob(os.path.join(validation_directory_speech, "*_mfcc.pickle"))

    val_feats_laugh = np.empty((len(val_files_laugh)), dtype=object)
    labels_frames_laugh = []
    for idx, file in enumerate(val_files_laugh):
        label = 1
        ses_id = file.split("/")[-1].strip('.pickle').split("_")[0]
        num_frames = int(file.split("/")[-1].strip('.pickle').split("_")[1])

        labels_frames_laugh.append([label] * num_frames)

        feat_file_de = "{0}/{1}_{2}_mfcc-de.pickle".format("/".join(file.split("/")[0:-1]), ses_id, num_frames)
        feat_file_de_de = "{0}/{1}_{2}_mfcc-de-de.pickle".format("/".join(file.split("/")[0:-1]), ses_id,
                                                                 num_frames)
        with open(file, 'rb') as f:
            temp = np.vstack(list(pickle.load(f)[0]))
            feats_laugh = feat_splice(np.transpose(temp), splice_context)

        with open(feat_file_de, 'rb') as f:
            temp = np.vstack(list(pickle.load(f)[0]))
            feats_laugh_de = feat_splice(np.transpose(temp), splice_context)

        with open(feat_file_de_de, 'rb') as f:
            temp = np.vstack(list(pickle.load(f)[0]))
            feats_laugh_de_de = feat_splice(np.transpose(temp), splice_context)

        val_feats_laugh[idx] = np.transpose(np.row_stack((feats_laugh, feats_laugh_de,feats_laugh_de_de)))

    val_feats_speech = np.empty((len(val_files_speech)), dtype=object)
    labels_frames_speech = []
    for idx, file in enumerate(val_files_speech):
        label = 0
        ses_id = file.split("/")[-1].strip('.pickle').split("_")[0]
        num_frames = int(file.split("/")[-1].strip('.pickle').split("_")[1])

        labels_frames_speech.append([label] * num_frames)

        feat_file_de = "{0}/{1}_{2}_mfcc-de.pickle".format("/".join(file.split("/")[0:-1]), ses_id, num_frames)
        feat_file_de_de = "{0}/{1}_{2}_mfcc-de-de.pickle".format("/".join(file.split("/")[0:-1]), ses_id,
                                                                 num_frames)
        with open(file, 'rb') as f:
            temp = np.vstack(list(pickle.load(f)[0]))
            feats_laugh = feat_splice(np.transpose(temp), splice_context)

        with open(feat_file_de, 'rb') as f:
            temp = np.vstack(list(pickle.load(f)[0]))
            feats_laugh_de = feat_splice(np.transpose(temp), splice_context)

        with open(feat_file_de_de, 'rb') as f:
            temp = np.vstack(list(pickle.load(f)[0]))
            feats_laugh_de_de = feat_splice(np.transpose(temp), splice_context)

        val_feats_speech[idx] = np.transpose(np.row_stack((feats_laugh, feats_laugh_de, feats_laugh_de_de)))

    #for epoch in range(training_params_DNN['num_epochs']):

    Generator = DataGenerator(**data_gen_params)

    training_gen = Generator.load_DataGenerators(train_feat_dir)
    val_gen = Generator.load_DataGenerators(val_feat_dir)

    model = train(model, training_gen,
                  val_gen,
                  epochs=training_params_DNN['num_epochs'],
                  train_steps_per_epoch=int(np.floor((307224+312780)/data_gen_params['batch_size'])),
                  val_steps_per_epoch=int(np.floor((58013+63944)/data_gen_params['batch_size']))
                  )

        # acc_laugh = 0
        # acc_speech = 0
        # for idx, val_feat in enumerate(val_feats_laugh):
        #
        #     out = model.predict(val_feat)
        #     target = np.array(labels_frames_laugh[idx], dtype=float)
        #
        #     out = [float(x >= 0.5) for y in list(out) for x in y]
        #     out = np.array(out)
        #     acc_laugh += np.sum(out == target)/out.shape[0]
        # for idx, val_feat in enumerate(val_feats_speech):
        #
        #     out = model.predict(val_feat)
        #     target = np.array(labels_frames_speech[idx], dtype=float)
        #
        #     out = [float(x >= 0.5) for y in list(out) for x in y]
        #     out = np.array(out)
        #     acc_speech += np.sum(out == target) / out.shape[0]
        #
        # print("laugh_acc = ", acc_laugh)
        # print("speech_acc = ", acc_speech)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    #feat_inp_dir = '/media/External_HD/tiles_audio/train_test/'
    feat_inp_dir = '/work/rperi/laugh/data/train_test/'
    

    speech_features_dir = '{0}/'.format(feat_inp_dir)

    #model_dir = '/media/External_HD/tiles_audio/models/'
    model_dir = '/work/rperi/laugh/models/'

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    main(speech_features_dir, model_dir)
