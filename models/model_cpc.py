import keras
from keras.models import Model
from keras.layers import Dense, Dropout, Input, BatchNormalization, UpSampling2D, Cropping2D, Flatten, GlobalMaxPooling2D, TimeDistributed, Layer
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras import metrics
from settings import *

from keras import backend as K
# K.clear_session()
K.set_image_data_format('channels_first')


class CPCLayer(keras.layers.Layer):

    ''' Computes dot product between true and predicted embedding vectors '''

    def __init__(self, **kwargs):
        super(CPCLayer, self).__init__(**kwargs)

    def call(self, inputs):

        # Compute dot product among vectors
        preds, y_encoded = inputs
        dot_product = K.mean(y_encoded * preds, axis=-1)
        dot_product = K.mean(dot_product, axis=-1, keepdims=True)  # average along the temporal dimension

        # Keras loss functions take probabilities
        dot_product_probs = K.sigmoid(dot_product)

        return dot_product_probs

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], 1)


def network_prediction(context, code_size, predict_terms):

    ''' Define the network mapping context to multiple embeddings '''

    outputs = []
    for i in range(predict_terms):
        outputs.append(Dense(units=code_size, activation="linear", name='z_t_{i}'.format(i=i))(context))

    if len(outputs) == 1:
        output = keras.layers.Lambda(lambda x: K.expand_dims(x, axis=1))(outputs[0])
    else:
        output = keras.layers.Lambda(lambda x: K.stack(x, axis=1))(outputs)

    return output

def network_autoregressive(x):

    ''' Define the network that integrates information along the sequence '''

    # x = keras.layers.GRU(units=256, return_sequences=True)(x)
    # x = keras.layers.BatchNormalization()(x)
    x = keras.layers.GRU(units=256, return_sequences=False, name='ar_context')(x)

    return x

def build_model(lrate=learning_rate_tl1, opt='adam', epochs=nb_epochs_tl1, decay=0.0, terms=terms, predict_terms=predict_terms):
    code_size = 256
    # Adam optimizer (Han2016)
    if opt == 'adam':
        optimizer = Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True, decay=decay)
        print('\n Building model: Using Adam Optimizer')
    # Stochastic Gradient Descent (Jordi2017)
    else:
        # select the optimizer to be used
        decay = lrate / epochs  # time based learning rate schedule (Brownlee)
        optimizer = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=True)
        print('\n Building model: Using SGD Optimizer')

    if feats_melspec:
        x_input = Input(shape=(1, num_bands, duration_segment))
    else:
        x_input = Input(shape=(1, 12, duration_segment))
    x = BatchNormalization(name='b1_b1')(x_input)
    x = Conv2D(32, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b1_c1')(x)
    x = BatchNormalization(name='b1_b2')(x)
    x = Conv2D(32, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b1_c2')(x)
    x = BatchNormalization(name='b1_b3')(x)
    x = MaxPooling2D(pool_size=(4, 1), data_format='channels_first', name='b1_mp1')(x)
    x = Dropout(0.25, name='b1_d1')(x)

    x = Conv2D(64, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b2_c1')(x)
    x = BatchNormalization(name='b2_b1')(x)
    x = Conv2D(64, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b2_c2')(x)
    x = BatchNormalization(name='b2_b2')(x)
    x = MaxPooling2D(pool_size=(4, 1), data_format='channels_first', name='b2_mp1')(x)
    x = Dropout(0.25, name='b2_d1')(x)

    x = Conv2D(128, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b3_c1')(x)
    x = BatchNormalization(name='b3_b1')(x)
    x = Conv2D(128, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b3_c2')(x)
    x = BatchNormalization(name='b3_b2')(x) 
    x = MaxPooling2D(pool_size=(2, 1), data_format='channels_first', name='b3_mp1')(x)
    x = Dropout(0.25, name='b3_d1')(x)

    x = Flatten()(x)

    x = Dense(256, activation='relu', name='b3_fc1')(x)
    x = BatchNormalization(name='b3_b3')(x) 
    encoded = Dense(code_size, name='encod')(x)

    encoder_mod = Model(x_input, encoded)

    # encoder_mod.summary()

    x_input = Input(shape=(terms, 1, num_bands, duration_segment))
    x_encoded = TimeDistributed(encoder_mod)(x_input)
    context = network_autoregressive(x_encoded)
    preds = network_prediction(context, code_size, predict_terms)

    y_input = Input(shape=(predict_terms, 1, num_bands, duration_segment))
    y_encoded = TimeDistributed(encoder_mod)(y_input)

    # loss 
    dot_product_probs = CPCLayer()([preds, y_encoded])

    cpc_model = Model(inputs=[x_input, y_input], outputs=dot_product_probs)

    cpc_model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['binary_accuracy'])
    return cpc_model


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    model = build_model()
    print(model.summary())
