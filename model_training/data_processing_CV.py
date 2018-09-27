import numpy as np
from sklearn import preprocessing
import random
from random import shuffle
import glob
import os
import pickle


def padding(feat_array, splice_context):
    feat_dimension, num_frames = feat_array.shape
    feat_array_padded = np.zeros((feat_dimension, num_frames + 2*splice_context))
    feat_array_padded[:, splice_context:num_frames + splice_context] = feat_array[:,:]
    for ctx in range(splice_context):
        feat_array_padded[:, ctx] = feat_array[:,0]
        feat_array_padded[:, num_frames + ctx] = feat_array[:, -1]

    return feat_array_padded


def feat_splice(feature_array, splice_context):

    feat_dim, num_frames = feature_array.shape
    feat_array_spliced = np.zeros((feat_dim*(splice_context*2 + 1), num_frames))

    feat_array_padded = padding(feature_array[:, :], splice_context)

    for frame in range(num_frames):
        for splice in range(splice_context*2 + 1):
            feat_array_spliced[splice*feat_dim: feat_dim + splice*feat_dim, frame] = feat_array_padded[:, frame + splice]

    return feat_array_spliced


def normalize(feature_array):
    feature_array_norm = np.empty(np.shape(feature_array))

    num_frames, feat_dim = np.shape(feature_array)

    for feat in range(feat_dim):
        feature_array_norm[:, feat] = feature_array[:, feat] - np.mean(feature_array[:, feat])

    return feature_array_norm


def get_data_shuffle(feat_dir):
    #random.seed(1234)
    feat_dir_laugh = "{0}/laugh/".format(feat_dir)
    feat_dir_speech = "{0}/speech/".format(feat_dir)


    # Get all available feature pickle files available
    feat_files_laugh = []
    feat_files_speech = []
    labels_laugh = []
    labels_speech = []

    speaker_list_laugh = glob.glob(os.path.join(feat_dir_laugh, "*"))
    speaker_list_speech = glob.glob(os.path.join(feat_dir_speech, "*"))

    for speaker in speaker_list_laugh:
        feat_files_laugh.append(glob.glob(os.path.join(feat_dir_laugh + speaker.split("/")[-1] + "/*_mfcc.pickle")))
        labels_laugh.append([1.0] * len(feat_files_laugh[-1]))
    feat_files_laugh = [x for y in feat_files_laugh for x in y]
    labels_laugh = [x for y in labels_laugh for x in y]

    num_laugh_segments = 0
    for feat_file in feat_files_laugh:
        num_laugh_segments += int(float(feat_file.split("/")[-1].strip('.pickle').split("_")[1]) / 100)

    for speaker in speaker_list_speech:
        feat_files_speech.append(glob.glob(os.path.join(feat_dir_speech + speaker.split("/")[-1] + "/*_mfcc.pickle")))
        labels_speech.append([0.0] * len(feat_files_speech[-1]))

    feat_files_speech = [x for y in feat_files_speech for x in y]
    labels_speech = [x for y in labels_speech for x in y]

    shuffle(feat_files_speech)
    num_speech_segments = 0
    for idx, feat_file in enumerate(feat_files_speech):
        # print(feat_file)
        num_speech_segments += int(float(feat_file.split("/")[-1].strip('.pickle').split("_")[1]) / 100)
        if num_speech_segments < num_laugh_segments:
            continue
        else:
            stop_idx = idx
            break
    feat_files_speech = feat_files_speech[:stop_idx]
    labels_speech = labels_speech[:stop_idx]

    feat_files = feat_files_laugh + feat_files_speech
    labels = labels_laugh + labels_speech
    # feat_files = [x for y in feat_files_laugh + feat_files_speech for x in y]
    # labels = [x for y in labels for x in y]

    temp = list(zip(feat_files, labels))
    shuffle(temp)
    feat_files, labels = zip(*temp)

    temp_feats = []
    temp_labels = []
    num_segments = 0
    total_segments = 0
    for idx, feat_file in enumerate(feat_files):
        #print("{0}/{1}".format(idx,len(feat_files)))
        label = labels[idx]

        ses_id = feat_file.split("/")[-1].strip('.pickle').split("_")[0]
        num_frames = int(feat_file.split("/")[-1].strip('.pickle').split("_")[1])

        num_segments = int(num_frames / 100)

        feat_file_de = "{0}/{1}_{2}_mfcc-de.pickle".format("/".join(feat_file.split("/")[0:-1]), ses_id, num_frames)
        feat_file_de_de = "{0}/{1}_{2}_mfcc-de-de.pickle".format("/".join(feat_file.split("/")[0:-1]), ses_id,
                                                                 num_frames)

        # feats[0] = [x for y in feats[0] for x in y]
        # for seg_idx in range(len(feats[0])):

        # mfcc features
        with open(feat_file, 'rb') as f:
            feats = pickle.load(f)

        s = [x for x in feats[0]]

        for s1 in s:
            if len(s1) != 100:
                print(feat_file)

        if num_frames % 100 == 0:
            temp_feats.append(s)
            temp_labels.append([label] * num_segments)
            total_segments += num_segments
        else:
            temp_feats.append(s[0:-1])
            temp_labels.append([label] * (num_segments) )
            total_segments += num_segments

    t = np.array([np.array(x) for x in [y for p in temp_feats for y in p]])
    if len(t.shape) == 3:
        if (t.shape[1] != 100) or (t.shape[2] != 14):
            print("dimension mismatch")
            return(np.array((0,0)), np.array((0,0)), np.array((0,0)))
    else:
        print("dimension mismatch")
        return (np.array((0, 0)), np.array((0, 0)), np.array((0, 0)))

    features = np.transpose(t[:, :, 0:13], (0, 2, 1))
    features = features.reshape(features.shape[0], features.shape[1], features.shape[2], 1)
    labels = np.array([np.array(x) for x in [y for p in temp_labels for y in p]])
    
    return (features, labels, total_segments)


def get_data_shuffle_DNN(feat_dir, splice_context):
    feat_dir_laugh = "{0}/laugh/".format(feat_dir)
    feat_dir_speech = "{0}/speech/".format(feat_dir)

    splice_context = splice_context

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
        labels_laugh.append([1.0] * len(feat_files_laugh[-1]))
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
        # print(feat_file)
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
    # feat_files = [x for y in feat_files_laugh + feat_files_speech for x in y]
    # labels = [x for y in labels for x in y]

    temp = list(zip(feat_files, labels))
    shuffle(temp)
    feat_files, labels = zip(*temp)

    #temp_feats = np.array([], dtype=object)
    #temp_labels = []
    for idx, feat_file in enumerate(feat_files):
        print("{0}/{1}".format(idx, len(feat_files)))
        label = labels[idx]

        ses_id = feat_file.split("/")[-1].strip('.pickle').split("_")[0]
        num_frames = int(feat_file.split("/")[-1].strip('.pickle').split("_")[1])

        labels_frames = [label] * num_frames

        #feat_file_de = "{0}/{1}_{2}_mfcc-de.pickle".format("/".join(feat_file.split("/")[0:-1]), ses_id, num_frames)
        #feat_file_de_de = "{0}/{1}_{2}_mfcc-de-de.pickle".format("/".join(feat_file.split("/")[0:-1]), ses_id,
        #                                                         num_frames)

        # feats[0] = [x for y in feats[0] for x in y]
        # for seg_idx in range(len(feats[0])):

        # mfcc features
        with open(feat_file, 'rb') as f:
            feats = pickle.load(f)
        # print(feat_file)
        temp = feats[0]
        temp = [x[0:13] for y in temp for x in y]
        feats_spliced = feat_splice(np.transpose(temp), splice_context)

        if idx == 0:
            temp_feats = feats_spliced
            temp_labels = np.array(labels_frames)
        else:
            temp_feats = np.column_stack((temp_feats, feats_spliced))
            temp_labels = np.concatenate((temp_labels, np.array(labels_frames)))
        #np.append(temp_feats, feats_spliced)
        #temp_labels.append(labels_frames)

    return (np.transpose(temp_feats), temp_labels)

