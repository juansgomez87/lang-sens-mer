import numpy as np
import importlib
import argparse
import time
import pdb
import os
from batch_generator import BatchGeneratorAutoenc
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import h5py
import random
from settings import *
from keras import backend as K


class Pretrainer:
    """ Trainer trains a convolutional neural network with a given model. The training data set
    must have already been extracted with *generate_dataset.py*. The balanced train-validation
    split is already performed during the data set generation, so the trainer only performs
    a randoms shuffling on every epoch. All of the parameters are inherited from *settings.py*.

    :param model: model to be trained with the selected data set.
    :param path_to_data: path to extracted h5py data sets.
    :type model: object
    :param path_to_data: str
    """

    def __init__(self, model, path_to_data, path_to_save):
        """Constructor method
        """
        self.start = time.time()
        self.model = model
        self.noise_factor = noise_factor
        bay_opt = False

        if str(model).find('model_over') > 0:
            # Data sets file paths
            TRAIN_TENSOR = os.path.join(path_to_data, 'train_dataset.h5py')
            train_tensor = h5py.File(TRAIN_TENSOR, 'r')
            train_keys = [_ for _ in train_tensor.keys()]
            self.train_gen = BatchGeneratorAutoenc(train_tensor, train_keys)

            VAL_TENSOR = os.path.join(path_to_data, 'val_dataset.h5py')
            val_tensor = h5py.File(VAL_TENSOR, 'r')
            val_keys = [_ for _ in val_tensor.keys()]
            self.valid_gen = BatchGeneratorAutoenc(val_tensor, val_keys)
        else:
            print('The selected model is not compatible with batch generator!')
            import sys
            sys.exit(0)

        print('*************\nUsing training dataset:', TRAIN_TENSOR)
        print('Using validation dataset:', VAL_TENSOR, '\n*************')
        self.path_to_save = path_to_save

        if bay_opt:
            import skopt
            from skopt import gp_minimize, forest_minimize
            from skopt.space import Real, Categorical, Integer
            from skopt.plots import plot_convergence
            from skopt.plots import plot_objective, plot_evaluations
            # from skopt.plots import plot_histogram, plot_objective_2D
            from skopt.utils import use_named_args
            import sys
            import tensorflow
            import matplotlib.pyplot as plt

            dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform',
                         name='learning_rate')
            dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")
            dimensions = [dim_learning_rate, dim_adam_decay]
            global best_accuracy
            best_accuracy = 0

            @use_named_args(dimensions=dimensions)
            def fitness(learning_rate, adam_decay):
                model_opt = model.build_model(lrate=learning_rate, decay=adam_decay)
                history = model_opt.fit_generator(self.train_gen,
                                                   epochs=3,
                                                   # class_weight=self.class_weights,
                                                   validation_data=self.valid_gen)
                acc = history.history['val_acc'][-1]
                print('Accuracy: ', acc)
                global best_accuracy
                if acc > best_accuracy:
                    best_accuracy = acc
                del model_opt
                tensorflow.compat.v1.reset_default_graph()
                K.clear_session()
                return -acc

            search_result = gp_minimize(func=fitness,
                                        dimensions=dimensions,
                                        acq_func='EI', # Expected Improvement.
                                        n_calls=40,
                                        random_state=0,
                                        n_jobs=-1,
                                        x0=[learning_rate_tl1, 1e-06])
            plot_convergence(search_result)
            # plt.show()

            # librispeech:
            # acc 0.624344 - lr 0.0009653265915661777 decay 4.244417355110645e-06
            # aishell
            # acc 0.63873 - lr 0.0017 decay 4.0810240705777585e-06
            print(sorted(zip(search_result.func_vals, search_result.x_iters)))

            sys.exit(0)




    def train(self, epochs, iter_num):
        """ This method trains the corresponding model and implements two callback
        functions: Early Stopping (patience = 5 epochs) and Model Checkpointing. The number of epochs and
        mini-batch size are inherited from *settings.py*. Additionally, the training history, weights, and 
        structure are stored in the MODEL_PATH.

        :param epoch: maximum number of epochs to train model.
        :param batch: mini-batch size.
        :param iter_num: number of training iterations to compensate for random initialization.
        :type epoch: int
        :type batch: int
        :type iter_num: int
        """
        # iterate over the number of experiments
        if feats_melspec:
            mod_type = 'spec'
        else:
            mod_type = 'chrom'
        for i in range(iter_num):
            print('****************\nTraining iteration number', i)
            K.clear_session()
            # build model
            model_iter = self.model.build_model(lrate=learning_rate_pre, decay=decay_pre)
            # print(self.model.summary())
            # model checkpointing with validation accuracy (Brownlee)
            weights_filename = os.path.join(self.path_to_save, '{}.it_{}.subset_{}.{}.best.hdf5'.format(args.model, str(i), subset_rate, mod_type))
            history_filename = os.path.join(self.path_to_save, '{}.it_{}.subset_{}.{}.npy'.format(args.model, str(i), subset_rate, mod_type))
            if os.path.exists(history_filename) == False:
                model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
                # serialize model to JSON
                model_json = model_iter.to_json()
                json_filename = os.path.join(self.path_to_save, '{}.it_{}.subset_{}.{}.json'.format(args.model, str(i), subset_rate, mod_type))
                with open(json_filename, 'w') as json_file:
                    json_file.write(model_json)
                # early stopping with validation loss, patience 3 (Han2016)
                early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
                reducrlr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

                history = model_iter.fit_generator(self.train_gen,
                                                   epochs=epochs,
                                                   # shuffle=True,
                                                   validation_data=self.valid_gen,
                                                   callbacks=[model_checkpoint, early_stopping, reducrlr])

                # save history for plots             
                np.save(history_filename, history.history)
            else:
                print('Model has been trained, just to new iteration!')
        print('Training finished, time to train: ',
              (time.time() - self.start) / 60)


if __name__ == '__main__':
    # for reproducibility
    np.random.seed(87)
    from tensorflow import set_random_seed
    set_random_seed(1987)
    # Usage python3 pretrain.py --dataset m/s --language e/m
    parser = argparse.ArgumentParser()

    parser.add_argument('-s',
                        '--speech',
                        help='Select language of data: english (e) or mandarin (m) or mix (x)',
                        action='store',
                        required=True,
                        dest='speech')
    parser.add_argument('-mod', '--model',
                        help='Select model from model_base, model_over1...',
                        action='store',
                        required=True,
                        dest='model')
    args = parser.parse_args()

    # import module
    try:
        mod_str = 'models.' + args.model
        model_module = importlib.import_module(mod_str)
    except:
        print('The chosen model does not exist')


    if args.speech == 'e':
        path_to_data = path_speech_feat_eng
        path_to_save = './models/speech/english'
        print('Path to data:', path_to_data)
    elif args.speech == 'm':
        path_to_data = path_speech_feat_man
        path_to_save = './models/speech/mandarin'
        print('Path to data:', path_to_data)
    elif args.speech == 'x':
        path_to_data = path_speech_feat_mix
        path_to_save = './models/speech/mix'
        print('Path to data:', path_to_data)

    # time process
    start = time.time()
    # initialize trainer
    trainer = Pretrainer(model_module, path_to_data, path_to_save)

    # train model
    trainer.train(nb_epochs_pre, 1)

    print('Total processing time: ', (time.time() - start) / 60, 'minutes.')
