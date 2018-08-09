import os
import glob
from random import shuffle
import pickle
import numpy as np
from data_processing import feat_splice
import keras


class DataGenerator(object):
    def __init__(self, inp_dim, target_dim, batch_size, splice_context):
        self.inp_dim = inp_dim
        self.target_dim = target_dim
        self.batch_size = int(batch_size)
        self.splice_context = splice_context

    def load_DataGenerators(self, feat_dir):

        while True:
            feat_dir_laugh = "{0}/laugh/".format(feat_dir)
            feat_dir_speech = "{0}/speech/".format(feat_dir)

            batch_size = self.batch_size
            splice_context = self.splice_context

            # Get all available feature pickle files available
            feat_files_laugh = []
            feat_files_speech = []
            labels_laugh = []
            labels_speech = []

            speaker_list_laugh = glob.glob(os.path.join(feat_dir_laugh, "*"))
            speaker_list_speech = glob.glob(os.path.join(feat_dir_speech, "*"))
            num_laugh_frames = 0
            for speaker in speaker_list_laugh:
                feat_files_laugh.append(glob.glob(os.path.join(feat_dir_laugh + speaker.split("/")[-1] + "/*_mfcc.pickle")))
                labels_laugh.append([1.0]*len(feat_files_laugh[-1]))
            feat_files_laugh = [x for y in feat_files_laugh for x in y]
            labels_laugh = [x for y in labels_laugh for x in y]

            for feat_file in feat_files_laugh:
                num_laugh_frames += int(feat_file.split("/")[-1].strip('.pickle').split("_")[1])

            num_speech_frames = 0
            for speaker in speaker_list_speech:
                feat_files_speech.append(glob.glob(os.path.join(feat_dir_speech + speaker.split("/")[-1] + "/*_mfcc.pickle")))
                labels_speech.append([0.0] * len(feat_files_speech[-1]))

            feat_files_speech = [x for y in feat_files_speech for x in y]
            labels_speech = [x for y in labels_speech for x in y]

            shuffle(feat_files_speech)
            for idx, feat_file in enumerate(feat_files_speech):
                #print(feat_file)
                num_speech_frames += int(feat_file.split("/")[-1].strip('.pickle').split("_")[1])
                if num_speech_frames < num_laugh_frames:
                    continue
                else:
                    stop_idx = idx
                    break
            feat_files_speech = feat_files_speech[:stop_idx]
            labels_speech = labels_speech[:stop_idx]

            feat_files = feat_files_laugh + feat_files_speech
            labels = labels_laugh + labels_speech
            #feat_files = [x for y in feat_files_laugh + feat_files_speech for x in y]
            #labels = [x for y in labels for x in y]

            temp = list(zip(feat_files, labels))
            shuffle(temp)
            feat_files, labels = zip(*temp)

            for idx, feat_file in enumerate(feat_files):

                label = labels[idx]

                ses_id = feat_file.split("/")[-1].strip('.pickle').split("_")[0]
                num_frames = int(feat_file.split("/")[-1].strip('.pickle').split("_")[1])

                labels_frames = [label]*num_frames

                feat_file_de = "{0}/{1}_{2}_mfcc-de.pickle".format("/".join(feat_file.split("/")[0:-1]), ses_id, num_frames)
                feat_file_de_de = "{0}/{1}_{2}_mfcc-de-de.pickle".format("/".join(feat_file.split("/")[0:-1]), ses_id, num_frames)


                #feats[0] = [x for y in feats[0] for x in y]
                #for seg_idx in range(len(feats[0])):

                # mfcc features
                with open(feat_file, 'rb') as f:
                    feats = pickle.load(f)
                #print(feat_file)
                temp = np.vstack(list(feats[0]))
                feats_spliced = feat_splice(np.transpose(temp), splice_context)

                # delta features
                with open(feat_file_de, 'rb') as f:
                    feats = pickle.load(f)
                #feats[0] = [x for y in feats[0] for x in y]
                temp = np.vstack(list(feats[0]))
                feats_spliced_de = feat_splice(np.transpose(temp), splice_context)

                # delta-delta features
                with open(feat_file_de_de, 'rb') as f:
                    feats = pickle.load(f)
                #feats[0] = [x for y in feats[0] for x in y]
                temp = np.vstack(list(feats[0]))
                feats_spliced_de_de = feat_splice(np.transpose(temp), splice_context)

                feat_dim, num_frames = feats_spliced.shape

                num_batches = int(np.floor(num_frames/batch_size))

                for batch in range(num_batches):
                    current_batch_feats = np.transpose(np.row_stack(
                        (feats_spliced[:, batch*batch_size: (batch+1)*batch_size],
                         feats_spliced_de[:, batch * batch_size: (batch + 1) * batch_size],
                         feats_spliced_de_de[:, batch * batch_size: (batch + 1) * batch_size])))
                    current_batch_labels = labels_frames[batch*batch_size: (batch+1)*batch_size]

                    yield current_batch_feats, current_batch_labels




