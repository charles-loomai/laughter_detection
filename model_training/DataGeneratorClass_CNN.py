import os
import glob
from random import shuffle
import pickle
import numpy as np
from data_processing import feat_splice, normalize
import keras

class DataGenerator(object):
    def __init__(self, batch_size):
        self.batch_size = int(batch_size)

    def load_DataGenerators(self, feat_dir):

        while True:
            feat_dir_laugh = "{0}/laugh/".format(feat_dir)
            feat_dir_speech = "{0}/speech/".format(feat_dir)

            batch_size = self.batch_size

            # Get all available feature pickle files available
            feat_files_laugh = []
            feat_files_speech = []
            labels_laugh = []
            labels_speech = []

            speaker_list_laugh = glob.glob(os.path.join(feat_dir_laugh, "*"))
            speaker_list_speech = glob.glob(os.path.join(feat_dir_speech, "*"))

            for speaker in speaker_list_laugh:
                feat_files_laugh.append(glob.glob(os.path.join(feat_dir_laugh + speaker.split("/")[-1] + "/*_mfcc.pickle")))
                labels_laugh.append([1.0]*len(feat_files_laugh[-1]))
            feat_files_laugh = [x for y in feat_files_laugh for x in y]
            labels_laugh = [x for y in labels_laugh for x in y]

            num_laugh_segments = 0
            for feat_file in feat_files_laugh:
                num_laugh_segments += int(float(feat_file.split("/")[-1].strip('.pickle').split("_")[1])/100)

            for speaker in speaker_list_speech:
                feat_files_speech.append(glob.glob(os.path.join(feat_dir_speech + speaker.split("/")[-1] + "/*_mfcc.pickle")))
                labels_speech.append([0.0] * len(feat_files_speech[-1]))

            feat_files_speech = [x for y in feat_files_speech for x in y]
            labels_speech = [x for y in labels_speech for x in y]

            shuffle(feat_files_speech)

            num_speech_segments = 0
            for idx, feat_file in enumerate(feat_files_speech):
                #print(feat_file)
                if feat_file.split("/")[-1].strip('.pickle').split("_")[1] == '0.0':
                    continue
                num_speech_segments += int(float(feat_file.split("/")[-1].strip('.pickle').split("_")[1])/100)
                if num_speech_segments < num_laugh_segments:
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

            num_segments = 0
            temp = []
            for idx, feat_file in enumerate(feat_files):

                label = labels[idx]
                print(feat_file)
                ses_id = feat_file.split("/")[-1].strip('.pickle').split("_")[0]
                num_frames = int(feat_file.split("/")[-1].strip('.pickle').split("_")[1])

                num_segments += int(num_frames/100)

                labels_segments = [label]*num_segments

                feat_file_de = "{0}/{1}_{2}_mfcc-de.pickle".format("/".join(feat_file.split("/")[0:-1]), ses_id, num_frames)
                feat_file_de_de = "{0}/{1}_{2}_mfcc-de-de.pickle".format("/".join(feat_file.split("/")[0:-1]), ses_id, num_frames)

                #feats[0] = [x for y in feats[0] for x in y]
                #for seg_idx in range(len(feats[0])):

                # mfcc features
                with open(feat_file, 'rb') as f:
                    feats = pickle.load(f)

                temp.append([x for x in feats[0]])


                if num_segments < batch_size:
                    continue
                #print(num_segments)
                t = np.array([np.array(x) for x in [y for p in temp for y in p]])
                
                num_batches = int(np.floor(num_segments/batch_size))
                
                temp = []
                num_segments = 0
                print(t.shape)
                
                for batch in range(num_batches):

                    current_batch_feats = np.transpose(t[batch * batch_size: (batch + 1) * batch_size, :, 0:13], (0, 2, 1))
                    current_batch_feats = current_batch_feats.reshape((batch_size,
                                                                       current_batch_feats.shape[1],
                                                                       current_batch_feats.shape[2],
                                                                       1))
                    current_batch_labels = labels_segments[batch * batch_size: (batch + 1) * batch_size]

                    yield current_batch_feats, current_batch_labels


