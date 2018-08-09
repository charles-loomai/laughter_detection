from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras import regularizers

def create_network_baseline(params):

    # Parameters
    input_shape = params['input_shape']
    num_FC_layers = params['num_FC_layers']
    num_FC_units = params['num_FC_units']

    dropout_rate = params['dropout_rate']
    Batch_norm_FLAG = params['Batch_norm_FLAG']
    Batch_norm_momentum = params['Batch_norm_momentum']

    l1_reg_weight = params['l1_regularizer_weight']

    # Layers
    input_layer = Input(input_shape)

    FC1 = Dense(num_FC_units[0],
                activation='relu',
                activity_regularizer=regularizers.l1(l1_reg_weight))(input_layer)
    FC1 = Dropout(dropout_rate)(FC1)
    if Batch_norm_FLAG:
        FC1 = BatchNormalization(momentum=Batch_norm_momentum)(FC1)

    FC2 = Dense(num_FC_units[1],
                activation='relu',
                activity_regularizer=regularizers.l1(l1_reg_weight))(FC1)
    FC2 = Dropout(dropout_rate)(FC2)
    if Batch_norm_FLAG:
        FC2 = BatchNormalization(momentum=Batch_norm_momentum)(FC2)

    output_layer = Dense(1, activation='sigmoid')(FC2)

    return Model(input_layer, output_layer, name='DNN_baseline')


def train(model, train_generator, val_generator, epochs=100, train_steps_per_epoch=100, val_steps_per_epoch=100):

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        verbose=2,
                        validation_data=val_generator,
                        validation_steps=val_steps_per_epoch)

    return model