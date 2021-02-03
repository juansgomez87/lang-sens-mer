from keras.models import Model
from keras.layers import Dense, Dropout, Input, BatchNormalization, UpSampling2D, Cropping2D, Flatten, GlobalMaxPooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
from keras.optimizers import Adam, SGD
from keras import metrics
from settings import *

from keras import backend as K
K.set_image_data_format('channels_first')


def build_model(lrate=learning_rate_tl1, opt='adam', epochs=nb_epochs_tl1, decay=0.0):
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
        input_lay = Input(shape=(1, num_bands, duration_segment))
    else:
        input_lay = Input(shape=(1, 12, duration_segment))
    x = BatchNormalization(name='b1_b1')(input_lay)
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

    # x = Conv2D(256, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b4_c1')(x)
    # x = BatchNormalization(name='b4_b1')(x)
    # x = Conv2D(256, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b4_c2')(x)
    # x = BatchNormalization(name='b4_b2')(x) 
    encoded = MaxPooling2D(pool_size=(2, 1), data_format='channels_first', name='encod')(x)

    x = UpSampling2D(size=(2, 1), data_format='channels_first', name='b5_u1')(encoded)
    # x = Conv2D(256, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b5_c1')(x)
    # x = BatchNormalization(name='b5_b1')(x)
    # x = Conv2D(256, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b5_c2')(x)
    # x = BatchNormalization(name='b5_b2')(x)
    # x = Dropout(0.25, name='b5_d1')(x)

    x = UpSampling2D(size=(2, 1), data_format='channels_first', name='b6_u1')(x)
    x = Conv2DTranspose(128, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b6_c1')(x)
    x = BatchNormalization(name='b6_b1')(x)
    x = Conv2DTranspose(128, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b6_c2')(x)
    x = BatchNormalization(name='b6_b2')(x)
    x = Dropout(0.25, name='b6_d1')(x)

    x = UpSampling2D(size=(4, 1), data_format='channels_first', name='b7_u1')(x)
    x = Conv2DTranspose(64, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b7_c1')(x)
    x = BatchNormalization(name='b7_b1')(x)
    x = Conv2DTranspose(64, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b7_c2')(x)
    x = BatchNormalization(name='b7_b2')(x)
    x = Dropout(0.25, name='b7_d1')(x)

    x = UpSampling2D(size=(4, 1), data_format='channels_first', name='b8_u1')(x)
    x = Conv2DTranspose(32, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='relu', name='b8_c1')(x)
    x = BatchNormalization(name='b8_b1')(x)
    x = Conv2DTranspose(1, (3, 3), kernel_initializer='glorot_uniform', padding='same', activation='sigmoid', name='b8_c2')(x)
    x = BatchNormalization(name='b8_b2')(x)

    model = Model(input_lay, x)

    model.compile(loss='mse', optimizer=optimizer, metrics=['acc'])
    return model


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    model = build_model()
    print(model.summary())
