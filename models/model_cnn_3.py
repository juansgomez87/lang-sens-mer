from keras.models import Model
from keras.layers import Dense, Dropout, Input, BatchNormalization, UpSampling2D, Cropping2D, Flatten, GlobalMaxPooling2D, Concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv2DTranspose
from keras.optimizers import Adam, SGD
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
    x = MaxPooling2D(pool_size=(4, 1), data_format='channels_first', name='b3_mp1')(x)
    x = Dropout(0.25, name='b3_d1')(x)

    # x = GlobalMaxPooling2D(data_format='channels_first')(x)
    x = Flatten(data_format='channels_first')(x)
    x = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b9_fc1')(x)
    x = Dropout(0.25, name='b9_d1')(x)
    x = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b9_fc2')(x)
    x = Dropout(0.25, name='b9_d2')(x)
    x = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b9_fc3')(x)

    # quadrants
    y1 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b10_fc1')(x)
    y1 = Dropout(0.25, name='b10_d1')(y1)
    y1 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b10_fc2')(y1)
    out_q = Dense(4, activation='softmax', name='quadrants')(y1)
    # arousal
    y2 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b11_fc1')(x)
    y2 = Dropout(0.25, name='b11_d1')(y2)
    y2 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b11_fc2')(y2)
    out_a = Dense(1, activation='sigmoid', name='arousal')(y2)
    # valence
    y3 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b12_fc1')(x)
    y3 = Dropout(0.25, name='b12_d1')(y3)
    y3 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b12_fc2')(y3)
    out_v = Dense(1, activation='sigmoid', name='valence')(y3)

    model = Model(input_lay, outputs=[out_q, out_a, out_v])

    loss = ['categorical_crossentropy',
            'binary_crossentropy',
            'binary_crossentropy']

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model


if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"

    model = build_model()
    print(model.summary())
