import numpy as np


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