import os
import glob
from random import shuffle
import pickle
import numpy as np
from data_processing import feat_splice
import keras

os.environ['KERAS_BACKEND'] = 'tensorflow'

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, labels, input_dim, batch_size, num_classes, splice_context):
        "Initialization"
        self.input_dim = input_dim
        self.batch_size = int(batch_size)
        self.num_classes = num_classes
        self.splice_context = splice_context

    def __len__(self):
        "Number of batches per epoch"

        num_frames = 0
        for ids in self.list_IDs:
            num_frames += int(ids.split("/")[-1].strip('.pickle').split("_")[1])

        return int(np.floor(num_frames / self.batch_size))

    def __getitem__(self, index):
        "Generate one batch of data"
        indexes = self.indexes[index*self.batch_size]

    def load_DataGenerators(self, feat_dir, speaker_list):

        while True:
            batch_size = self.batch_size
            splice_context = self.splice_context

            # Get all available feature pickle files available
            feat_files = []
            for speaker in speaker_list:
                feat_files.append(glob.glob(os.path.join(feat_dir + speaker + "/*_mfcc.pickle")))

            shuffle(feat_files)

            for feat_file in feat_files:

                ses_id = feat_file.split("/")[-1].strip('.pickle').split("_")[0]
                num_frames = feat_file.split("/")[-1].strip('.pickle').split("_")[1]

                feat_file_de = "{0}/{1}_{2}_mfcc-de.pickle".format("/".join(feat_file.split("/")[0:-1]), ses_id, num_frames)
                feat_file_de_de = "{0}/{1}_{2}_mfcc-de-de.pickle".format("/".join(feat_file.split("/")[0:-1]), ses_id, num_frames)

                with open(feat_file, 'rb') as f:
                    feats = pickle.load(f)
                for seg_idx in range(len(feats[0])):

                    # mfcc features
                    feats_spliced = feat_splice(np.array(feats[0][seg_idx]), splice_context)

                    # delta features
                    with open(feat_file_de, 'rb') as f:
                        feats = pickle.load(f)
                    feats_spliced_de = feat_splice(np.array(feats[0][seg_idx]), splice_context)

                    # delta-delta features
                    with open(feat_file_de_de, 'rb') as f:
                        feats = pickle.load(f)
                    feats_spliced_de_de = feat_splice(np.array(feats[0][seg_idx]), splice_context)

                    feat_dim, num_frames = feats_spliced.shape

                    num_batches = int(np.floor(num_frames/batch_size))

                    for batch in range(num_batches):
                        current_batch = np.row_stack(
                            (feats_spliced[:, batch*batch_size: (batch+1)*batch_size],
                             feats_spliced_de[:, batch * batch_size: (batch + 1) * batch_size],
                             feats_spliced_de_de[:, batch * batch_size: (batch + 1) * batch_size]))

                        yield current_batch




