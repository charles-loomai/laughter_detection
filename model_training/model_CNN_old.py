from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization, Conv2D, Flatten
from keras import regularizers


def create_network_baseline(params):

    # Parameters
    input_shape = params['input_shape']

    num_conv_layers = params['num_conv_layers']
    kernels = params['kernels']
    kernel_size = params['kernel_size']


    num_FC_layers = params['num_FC_layers']
    num_FC_units = params['num_FC_units']

    dropout_rate = params['dropout_rate']
    Batch_norm_FLAG = params['Batch_norm_FLAG']
    Batch_norm_momentum = params['Batch_norm_momentum']

    l1_reg_weight = params['l1_regularizer_weight']

    # Layers
    input_layer = Input(input_shape)

    conv1 = Conv2D(kernels[0], kernel_size=kernel_size[0],
                   activity_regularizer=regularizers.l2(l1_reg_weight),
                   batch_size=32)(input_layer)

    if Batch_norm_FLAG:
        conv1 = BatchNormalization(momentum=Batch_norm_momentum)(conv1)

    conv2 = Conv2D(kernels[1], kernel_size=kernel_size[1],
                   activity_regularizer=regularizers.l2(l1_reg_weight))(conv1)

    if Batch_norm_FLAG:
        conv2 = BatchNormalization(momentum=Batch_norm_momentum)(conv2)

    conv_flat = Flatten()(conv2)

    FC1 = Dense(num_FC_units[0],
                activation='relu')(conv_flat)#,
                #activity_regularizer=regularizers.l2(l1_reg_weight))(FC1)
    FC1 = Dropout(dropout_rate)(FC1)

    if Batch_norm_FLAG:
        FC1 = BatchNormalization(momentum=Batch_norm_momentum)(FC1)

    FC2 = Dense(num_FC_units[1],
                activation='relu',
                activity_regularizer=regularizers.l2(l1_reg_weight))(FC1)
    FC2 = Dropout(dropout_rate)(FC2)

    if Batch_norm_FLAG:
        FC2 = BatchNormalization(momentum=Batch_norm_momentum)(FC2)

    output_layer = Dense(1, activation='sigmoid')(FC2)

    return Model(input_layer, output_layer, name='CNN')


def train(model, train_generator, val_generator, epochs=100, train_steps_per_epoch=100, val_steps_per_epoch=100):

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        verbose=2,
                        validation_data=val_generator,
                        validation_steps=val_steps_per_epoch)

    return model