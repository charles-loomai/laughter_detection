from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization


def create_network_baseline(input_shape, params):

    # Parameters
    num_FC_layers = 2
    num_FC_units = 400

    dropout_rate = 0.2
    Batch_norm_FLAG = True
    Batch_norm_momentum = 0.99

    # Layers
    input_layer = Input(input_shape)

    FC1 = Dense(num_FC_units, activation='relu')(input_layer)
    FC1 = Dropout(dropout_rate)(FC1)
    if Batch_norm_FLAG:
        FC1 = BatchNormalization(momentum=Batch_norm_momentum)(FC1)

    FC2 = Dense(num_FC_units, activation='relu')(FC1)
    FC2 = Dropout(dropout_rate)(FC2)
    if Batch_norm_FLAG:
        FC2 = BatchNormalization(momentum=Batch_norm_momentum)(FC2)

    output_layer = Dense(2, activation='softmax')(FC2)

    return Model(input_layer, output_layer, name='DNN_baseline')
