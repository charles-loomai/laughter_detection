from keras.models import Model
from keras.layers import Dense, Input, Dropout, BatchNormalization
from keras import regularizers
from keras.callbacks import Callback


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
                activation='relu')(input_layer)#,
                #activity_regularizer=regularizers.l2(l1_reg_weight))(input_layer)
    FC1 = Dropout(dropout_rate)(FC1)
    if Batch_norm_FLAG:
        FC1 = BatchNormalization(momentum=Batch_norm_momentum)(FC1)

    FC2 = Dense(num_FC_units[1],
                activation='relu')(FC1)#,
                #activity_regularizer=regularizers.l2(l1_reg_weight))(FC1)
    FC2 = Dropout(dropout_rate)(FC2)
    if Batch_norm_FLAG:
        FC2 = BatchNormalization(momentum=Batch_norm_momentum)(FC2)

    FC3 = Dense(num_FC_units[2],
                activation='relu',
                activity_regularizer=regularizers.l2(l1_reg_weight))(FC2)
    FC3 = Dropout(dropout_rate)(FC3)
    if Batch_norm_FLAG:
        FC3 = BatchNormalization(momentum=Batch_norm_momentum)(FC3)

    FC4 = Dense(num_FC_units[3],
                activation='relu',
                activity_regularizer=regularizers.l2(l1_reg_weight))(FC3)
    FC4 = Dropout(dropout_rate)(FC4)
    if Batch_norm_FLAG:
        FC4 = BatchNormalization(momentum=Batch_norm_momentum)(FC4)

    output_layer = Dense(1, activation='sigmoid')(FC4)

    return Model(input_layer, output_layer, name='DNN_baseline')


def train(model, train_generator, val_generator, epochs=100, train_steps_per_epoch=100, val_steps_per_epoch=100):

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_steps_per_epoch,
                        epochs=epochs,
                        verbose=2,
                        validation_data=val_generator,
                        validation_steps=val_steps_per_epoch)

    return model


def train_noDataGen(model, train_data, val_data, epochs=100, batch_size=32):

    train_feats = train_data[0]
    train_labels = train_data[1]

    model.fit(train_feats, train_labels, batch_size=batch_size, epochs=epochs, callbacks=[ValCallback(val_data)])
#, steps_per_epoch=train_steps_per_epoch)


class ValCallback(Callback):
    def __init__(self, val_data):
        self.val_data = val_data

    def on_epoch_end(self, epoch, logs={}):
        x, y = self.val_data
        loss, acc = self.model.evaluate(x, y, verbose=0)
        print('\nValidation loss: {}, acc: {}\n'.format(loss, acc))
