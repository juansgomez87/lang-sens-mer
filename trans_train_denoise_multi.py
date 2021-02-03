import numpy as np
import importlib
import seaborn as sns
import argparse
import pandas as pd
import time
import pdb
import os
from batch_generator import BatchGeneratorClassMulti
from keras import metrics
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential, model_from_json, load_model, Model
from keras.optimizers import SGD, RMSprop, Adam
from keras.layers import Dense, Dropout, Flatten, GlobalMaxPooling2D, Conv2D, BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import h5py
from settings import *
from keras import backend as K
K.clear_session()

class TransTrainer():
    def __init__(self,
                 path_model_load,
                 path_to_save,
                 path_to_data,
                 path_to_anno,
                 model_name,
                 rel_flag):
        """Constructor method
        """
        self.start = time.time()
        self.path_to_anno = path_to_anno
        self.model_name = model_name
        self.rel_flag = rel_flag

        bay_opt = False

        if self.model_name.find('model_cnn') < 0:
            # load models
            self.j_f, w_f = self.model_selector(path_model_load, self.model_name)
            self.model = self.load_pretrained_model(self.j_f, w_f)
        else:
            mod_str = 'models.' + self.model_name
            model_mod = importlib.import_module(mod_str)
            self.model = model_mod.build_model(lrate=learning_rate_tl1)

        # load data 
        # load class weights
        self.class_weights = np.load(os.path.join(path_to_data, 'class_weights.npy'), allow_pickle=True).tolist()

        # Data sets file paths
        TRAIN_TENSOR = os.path.join(path_to_data, 'train_dataset.h5py')
        train_tensor = h5py.File(TRAIN_TENSOR, 'r')
        train_keys = [_ for _ in train_tensor.keys()]
        self.train_gen = BatchGeneratorClassMulti(train_tensor, train_keys, self.path_to_anno)

        VAL_TENSOR = os.path.join(path_to_data, 'val_dataset.h5py')
        val_tensor = h5py.File(VAL_TENSOR, 'r')
        val_keys = [_ for _ in val_tensor.keys()]
        self.valid_gen = BatchGeneratorClassMulti(val_tensor, val_keys, self.path_to_anno)

        TEST_TENSOR = os.path.join(path_to_data, 'test_dataset.h5py')
        test_tensor = h5py.File(TEST_TENSOR, 'r')
        test_keys = [_ for _ in test_tensor.keys()]
        self.test_gen = BatchGeneratorClassMulti(test_tensor, test_keys, self.path_to_anno, shuffle=False)

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
            import tensorflow
            import sys

            dim_learning_rate = Real(low=1e-6, high=1e-1, prior='log-uniform',
                         name='learning_rate')
            dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")
            dimensions = [dim_learning_rate, dim_adam_decay]
            global best_accuracy
            best_accuracy = 0

            @use_named_args(dimensions=dimensions)
            def fitness(learning_rate, adam_decay):
                model_opt = model_mod.build_model(lrate=learning_rate, decay=adam_decay)
                history = model_opt.fit_generator(self.train_gen,
                                                   epochs=3,
                                                   class_weight=self.class_weights,
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
            plt.show()
            # training from scratch model_cnn
            # acc 0.5194 lr 9.353972456152052e-05-0.00021, adam dec 1e-06, data set 4q
            # acc 0.5597 lr 0.0001-0.0002726571183309215, adam dec 0.00963679072828914- 0.0001, data set ch
            print(sorted(zip(search_result.func_vals, search_result.x_iters)))

            sys.exit(0)


    def model_selector(self, path, model_name):
        # sel = input('Select model to load [0,1,2,3,4,5]:\n')
        sel = '0'
        sel_txt = 'it_' + sel
        if feats_melspec:
            files = [os.path.join(path, f) for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and f.find(sel_txt) > 0 and f.find(model_name) == 0 and f.find('spec') >= 0)]
        else:
            files = [os.path.join(path, f) for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and f.find(sel_txt) > 0 and f.find(model_name) == 0 and f.find('chrom') >= 0)]
        weights_filename = [_ for _ in files if _.endswith('.hdf5')][0]
        json_filename = [_ for _ in files if _.endswith('.json')][0]
        return json_filename, weights_filename

    def load_pretrained_model(self, json_file, weight_file):
        """ This method loads the pretrained models, loads the 
        weights and adds the new layers"""
        # load model
        j_f = open(json_file, 'r')
        loaded_model = j_f.read()
        j_f.close()
        model = model_from_json(loaded_model)
        # load weights
        model = load_model(weight_file)

        self.loss = ['categorical_crossentropy',
                     'binary_crossentropy',
                     'binary_crossentropy']

        # freeze layers and remove decoder
        cnt = 0
        for layer in model.layers:
            layer.trainable = False
            if layer.name.find('b5') >= 0 or layer.name.find('b6') >= 0 or layer.name.find('b7') >= 0 or layer.name.find('b8') >= 0:
                cnt += 1

        for i in range(cnt):
            model.layers.pop()

        # add classifier
        # x = GlobalMaxPooling2D(data_format='channels_first')(model.layers[-1].output)
        x = Flatten(data_format='channels_first')(model.layers[-1].output)
        x = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b9_fc1')(x)
        x = Dropout(0.25, name='b9_d1')(x)
        x = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b9_fc2')(x)
        x = Dropout(0.25, name='b9_d2')(x)
        x = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b9_fc3')(x)

        # quadrants
        y1 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b10_fc1')(x)
        y1 = Dropout(0.25, name='b10_d1')(y1)
        y1 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b10_fc2')(y1)
        y1 = Dense(4, activation='softmax', name='quadrants')(y1)

        # arousal
        y2 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b11_fc1')(x)
        y2 = Dropout(0.25, name='b11_d1')(y2)
        y2 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b11_fc2')(y2)
        y2 = Dense(2, activation='softmax', name='arousal')(y2)

        # valence
        y3 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b12_fc1')(x)
        y3 = Dropout(0.25, name='b12_d1')(y3)
        y3 = Dense(512, activation='relu', kernel_initializer='glorot_uniform', name='b12_fc2')(y3)
        y3 = Dense(2, activation='softmax', name='valence')(y3)

        model = Model(model.input, outputs=[y1, y2, y3])
        # optimizer

        opt = Adam(lr=learning_rate_tl1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, amsgrad=True, decay=decay_tl)
        model.compile(loss=self.loss, optimizer=opt, metrics=['acc'])
        # print(model.metrics_names)
        # model.summary()
        # pdb.set_trace()
        return model

    def train(self, epochs1, epochs2, iter_num):
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
        if self.model_name.find('model_cnn') > 0:
            # for training fast
            self.rel_flag = False
        for i in range(iter_num):
            print('****************\nTraining iteration number', i)
            # build model
            model_iter = self.model
            if feats_melspec:
                mod_type = 'spec'
            else:
                mod_type = 'chrom'

            if self.rel_flag:
                file_suffix = ".multitask.{}.it_{}.lr_{}".format(mod_type, i, learning_rate_tl2)
            else:
                file_suffix = ".multitask.{}.it_{}.lr_{}".format(mod_type, i, learning_rate_tl1)
            json_filename = os.path.join(self.path_to_save, self.model_name + file_suffix + '.json')
            weights_filename = os.path.join(self.path_to_save, self.model_name + file_suffix + '.hdf5')
            history_filename = os.path.join(self.path_to_save, self.model_name + file_suffix + '.npy')
            
            if not os.path.exists(history_filename):
                model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_quadrants_acc', verbose=1, save_best_only=True, mode='max')
                # serialize model to JSON
                model_json = model_iter.to_json()
                with open(json_filename, 'w') as json_file:
                    json_file.write(model_json)
                # early stopping with validation loss, patience 3 (Han2016)
                early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
                reducrlr = ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.5, verbose=1)

                history = model_iter.fit_generator(self.train_gen,
                                                   epochs=epochs1,
                                                   class_weight=self.class_weights,
                                                   validation_data=self.valid_gen,
                                                   shuffle=True,
                                                   callbacks=[model_checkpoint, early_stopping, reducrlr])
                # print(history.history)
                ## release weights and train with dev data set
                if self.rel_flag:
                    print('****************\nReleasing weights')
                    # un freeze weights
                    for layer in model_iter.layers:
                        layer.trainable = True
                    #  change early stopping
                    early_stopping = EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto')
                    model_checkpoint = ModelCheckpoint(weights_filename, monitor='val_quadrants_acc', verbose=1, save_best_only=True, mode='max')
                    # optimizer
                    opt = Adam(lr=learning_rate_tl2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=decay_tl)
                    model_iter.compile(loss=self.loss, optimizer=opt, metrics=['acc'])
                    history_rel = model_iter.fit_generator(self.train_gen,
                                                           epochs=epochs2,
                                                           class_weight=self.class_weights,
                                                           validation_data=self.valid_gen,
                                                           shuffle=True,
                                                           callbacks=[model_checkpoint, early_stopping, reducrlr])
                    for key in history.history.keys():
                        history.history[key].extend(history_rel.history[key])
                # print(history.history)
                np.save(history_filename, history.history)
                # test and save results
                self.save_results(model_iter, json_filename)
                print('Training finished, time to train: ', (time.time() - self.start) / 60)
            else:
                print('Model has been trained, jump to new iteration!')
                # self.save_results(model_iter, json_filename)


    def plot_confusion_matrix(self, path, matrix, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

        # Compute confusion matrix
        cm = matrix
        # # Only use the labels that appear in the data
        # classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        # print(cm)
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 6)
        cm = pd.DataFrame(cm, index=classes, columns=classes)
        ax = sns.heatmap(cm, cmap="YlGnBu", cbar=False, ax=ax, annot=True, square=True)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        # ax.xaxis.set_ticklabels(classes)
        # ax.yaxis.set_ticklabels(classes)

        # fig, ax = plt.subplots()
        # # fig.set_size_inches(6, 8)
        # im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        # # ax.figure.colorbar(im, ax=ax)
        # # We want to show all ticks...
        # ax.set(xticks=np.arange(cm.shape[1]),
        #        yticks=np.arange(cm.shape[0]),
        #        # ... and label them with the respective list entries
        #        xticklabels=classes, yticklabels=classes,
        #        # title=title,
        #        ylabel='True label',
        #        xlabel='Predicted label')

        # # Rotate the tick labels and set their alignment.
        # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        #          rotation_mode="anchor")

        # # Loop over data dimensions and create text annotations.
        # fmt = '.2f' if normalize else 'd'
        # thresh = cm.max() / 2.
        # for i in range(cm.shape[0]):
        #     for j in range(cm.shape[1]):
        #         ax.text(j, i, format(cm[i, j], fmt),
        #                 ha="center", va="center",
        #                 color="white" if cm[i, j] > thresh else "black")
        # fig.tight_layout()
        fig.savefig(path, bbox_inches='tight')

    def save_results(self, model, filename):
        plot_predictions = True
        # filenames
        txt_file = filename.replace('.json', '.txt')
        eps_file = filename.replace('.json', '.eps')
        
        # predict and plot
        print('****************\nMaking predictions...')
        y_pred_quads, y_pred_arousal, y_pred_valence = model.predict_generator(self.test_gen, verbose=1)

        y_test_quads = [self.test_gen.__getitem__(_)[1][0] for _ in range(self.test_gen.__len__())]
        y_test_arousal = [self.test_gen.__getitem__(_)[1][1] for _ in range(self.test_gen.__len__())]
        y_test_valence = [self.test_gen.__getitem__(_)[1][2] for _ in range(self.test_gen.__len__())]

        
        # test = list(zip(self.test_gen.anno_list_quads, self.test_gen.anno_list_arousal, self.test_gen.anno_list_valence))
        # print(test)
        # print([y_test_quads[_][0], y_test_arousal[_][0], y_test_valence[_][0] for _ in range(10)])
        
        y_test_quads = np.vstack(y_test_quads)

        if self.model_name.find('model_cnn_3') == 0:
            y_test_arousal = np.expand_dims(np.hstack(y_test_arousal), axis=1)
            y_test_valence = np.expand_dims(np.hstack(y_test_arousal), axis=1)
        else:
            y_test_arousal = np.vstack(y_test_arousal)
            y_test_valence = np.vstack(y_test_valence)

        y_pred_max_quads = np.zeros(y_pred_quads.shape)
        y_pred_max_arousal = np.zeros(y_pred_arousal.shape)
        y_pred_max_valence = np.zeros(y_pred_valence.shape)
        for i in range(y_pred_max_quads.shape[0]):
            y_pred_max_quads[i, np.argmax(y_pred_quads, axis=1)[i]] = np.max(y_pred_quads, axis=1)[i]
            if self.model_name.find('model_cnn_3') < 0:
                y_pred_max_arousal[i, np.argmax(y_pred_arousal, axis=1)[i]] = np.max(y_pred_arousal, axis=1)[i]
                y_pred_max_valence[i, np.argmax(y_pred_valence, axis=1)[i]] = np.max(y_pred_valence, axis=1)[i]

        # pdb.set_trace()
        # calculate metrics: recall, precision and f-score
        report_quads = classification_report(y_test_quads, np.ceil(y_pred_max_quads))
        if self.model_name.find('model_cnn_3') < 0:
            report_valence = classification_report(y_test_arousal, np.ceil(y_pred_max_arousal))
            report_arousal = classification_report(y_test_valence, np.ceil(y_pred_max_valence))
        else:
            report_valence = classification_report(y_test_arousal, np.round(y_pred_arousal))
            report_arousal = classification_report(y_test_valence, np.round(y_pred_valence))
        # calculate confusion matrices
        y_test_int_quads = np.argmax(y_test_quads, axis=1)
        if self.model_name.find('model_cnn_3') < 0:
            y_test_int_arousal = np.argmax(y_test_arousal, axis=1)
            y_test_int_valence = np.argmax(y_test_valence, axis=1)
        else:
            y_test_int_arousal = np.round(y_test_arousal)
            y_test_int_valence = np.round(y_test_valence)

        y_pred_int_quads = np.argmax(np.ceil(y_pred_max_quads), axis=1)
        if self.model_name.find('model_cnn_3') < 0:
            y_pred_int_arousal = np.argmax(np.ceil(y_pred_max_arousal), axis=1)
            y_pred_int_valence = np.argmax(np.ceil(y_pred_max_valence), axis=1)
        else:
            y_pred_int_arousal = np.round(y_pred_max_arousal)
            y_pred_int_valence = np.round(y_pred_max_valence)

        matrix_quads = confusion_matrix(y_test_int_quads, y_pred_int_quads)
        matrix_arousal = confusion_matrix(y_test_int_arousal, y_pred_int_arousal)
        matrix_valence = confusion_matrix(y_test_int_valence, y_pred_int_valence)

        matrix_norm_quads = matrix_quads.astype('float') / matrix_quads.sum(axis=1)[:, np.newaxis]
        matrix_norm_arousal = matrix_arousal.astype('float') / matrix_arousal.sum(axis=1)[:, np.newaxis]
        matrix_norm_valence = matrix_valence.astype('float') / matrix_valence.sum(axis=1)[:, np.newaxis]

        print('****************\nEvaluating model...')
        evaluate = model.evaluate_generator(self.test_gen, verbose=1)
        eval_res = ['{}: {}\n'.format(a, b) for a,b in zip(model.metrics_names, evaluate)]
        eval_res = ''.join(eval_res)

        txt_info = '{} \nTest results:\n{}\n\n---Quads---\n{}\n{}\n\n---Arousal---\n{}\n{}\n\n---Valence---\n{}\n{}'.format(filename,
                                                                               eval_res,
                                                                               report_quads,
                                                                               matrix_norm_quads,
                                                                               report_arousal,
                                                                               matrix_norm_arousal,
                                                                               report_valence,
                                                                               matrix_norm_valence)
        with open(txt_file, 'w') as text_file:
            print(txt_info, file=text_file)


        classes_quads = ['V+A+', 'V-A+', 'V-A-', 'V+A-']
        classes_arousal = ['A-', 'A+']
        classes_valence = ['V-', 'V+']

        eps_file_quads = eps_file.replace('.eps', '.quads.eps')
        eps_file_arousal = eps_file.replace('.eps', '.arousal.eps')
        eps_file_valence = eps_file.replace('.eps', '.valence.eps')

        self.plot_confusion_matrix(eps_file_quads, matrix_quads, classes_quads, normalize=True)
        self.plot_confusion_matrix(eps_file_arousal, matrix_arousal, classes_arousal, normalize=True)
        self.plot_confusion_matrix(eps_file_valence, matrix_valence, classes_valence, normalize=True)
        if plot_predictions:
            jpg_file = filename.replace('.json', '.jpg')
            plt.subplot(321)
            plt.imshow(y_test_quads, aspect='auto', interpolation='nearest')
            plt.title('Test labels - Quads')
            plt.subplot(322)
            plt.imshow(np.ceil(y_pred_max_quads), aspect='auto', interpolation='nearest')
            plt.title('Pred labels - Quads')
            plt.subplot(323)
            plt.imshow(y_test_arousal, aspect='auto', interpolation='nearest')
            plt.title('Test labels - Arousal')
            plt.subplot(324)
            plt.imshow(np.ceil(y_pred_max_arousal), aspect='auto', interpolation='nearest')
            plt.title('Pred labels - Arousal')
            plt.subplot(325)
            plt.imshow(y_test_valence, aspect='auto', interpolation='nearest')
            plt.title('Test labels - Valence')
            plt.subplot(326)
            plt.imshow(np.ceil(y_pred_max_valence), aspect='auto', interpolation='nearest')
            plt.title('Pred labels - Valence')
            plt.tight_layout()
            plt.savefig(jpg_file)

if __name__ == "__main__":
    # import os
    # os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    # print('------\n-----\nTRAINING WITH CPU!!!!\n------\n-----\n')

    # for reproducibility
    np.random.seed(87)
    from tensorflow import set_random_seed
    set_random_seed(87)
    
    # Usage python3 trans_train.py --pretrain e/m --dataset e/m --model [model_base, model_over_1]
    parser = argparse.ArgumentParser()
    parser.add_argument('-s',
                        '--speech',
                        help='Select from pretrained models on speech in english (e) or mandarin (m)',
                        action='store',
                        required=True,
                        dest='speech')
    parser.add_argument('-m',
                        '--music',
                        help='Select music of data for transfer learning: english (e) or mandarin (m)',
                        action='store',
                        required=True,
                        dest='music')
    parser.add_argument('-mod', '--model',
                        help='Select model from model_base, model_over_1...',
                        action='store',
                        required=True,
                        dest='model')
    parser.add_argument('-rel', '--release',
                        help='Unfreeze weight [y - 2 step TL] or [n - 1 step TL]...',
                        action='store',
                        required=True,
                        dest='release')

    args = parser.parse_args()

    if args.speech == 'e' and args.music == 'e':
        path_to_save = 'models_trans/speech_eng_2_music_eng'
        path_model_load = 'models/speech/english'
        path_to_data = path_music_feat_eng
        path_to_anno = path_music_anno_eng
        print('******\n INTRALINGUISTIC CASE: eng 2 eng\n')
    elif args.speech == 'e' and args.music == 'm':
        path_to_save = 'models_trans/speech_eng_2_music_man'
        path_model_load = 'models/speech/english'
        path_to_data = path_music_feat_man
        path_to_anno = path_music_anno_man
        print('******\n CROSSLINGUISTIC CASE: eng 2 man\n')
    elif args.speech == 'm' and args.music == 'm':
        path_to_save = 'models_trans/speech_man_2_music_man'
        path_model_load = 'models/speech/mandarin'
        path_to_data = path_music_feat_man
        path_to_anno = path_music_anno_man
        print('******\n INTRALINGUISTIC CASE: man 2 man\n')
    elif args.speech == 'm' and args.music == 'e':
        path_to_save = 'models_trans/speech_man_2_music_eng'
        path_model_load = 'models/speech/mandarin'
        path_to_data = path_music_feat_eng
        path_to_anno = path_music_anno_eng
        print('******\n CROSSLINGUISTIC CASE: man 2 eng\n')
    elif args.speech == 'x' and args.music == 'm':
        path_to_save = 'models_trans/speech_mix_2_music_man'
        path_model_load = 'models/speech/mix'
        path_to_data = path_music_feat_man
        path_to_anno = path_music_anno_man
        print('******\n MIX CASE: mix 2 man\n')
    elif args.speech == 'x' and args.music == 'e':
        path_to_save = 'models_trans/speech_mix_2_music_eng'
        path_model_load = 'models/speech/mix'
        path_to_data = path_music_feat_eng
        path_to_anno = path_music_anno_eng
        print('******\n MIX CASE: mix 2 eng\n')


    if args.model.find('model_cnn') >= 0 and args.music == 'e':
        path_to_save = 'models/music/english'
        path_model_load = None
        path_to_data = path_music_feat_eng
        path_to_anno = path_music_anno_eng
        print('******\n TRAIN FROM SCRATCH: eng\n')
    elif args.model.find('model_cnn') >= 0 and args.music == 'm':
        path_to_save = 'models/music/mandarin'
        path_model_load = None
        path_to_data = path_music_feat_man
        path_to_anno = path_music_anno_man
        print('******\n TRAIN FROM SCRATCH: man\n')   


    if args.release == 'y':
        rel_flag = True
    else:
        rel_flag = False

    # instanciate trainer
    trainer = TransTrainer(path_model_load, path_to_save, path_to_data, path_to_anno, args.model, rel_flag)

    # train model
    trainer.train(nb_epochs_tl1, nb_epochs_tl2, 1)