import h5py
import numpy as np
from settings import *
from keras.utils import Sequence, to_categorical
import argparse
import pandas as pd
import os
import pdb
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE, KMeansSMOTE
from collections import Counter

class BatchGeneratorAutoenc(Sequence):
    """ Generates data for Keras
    """
    def __init__(self,
                 file_data,
                 list_ids,
                 batch_size=batch_size_pre,
                 dim=(1, num_bands, duration_segment),
                 n_channels=1,
                 shuffle=True,
                 noise_factor=noise_factor):
        """Constructor method
        """
        self.file_data = file_data
        self.list_ids = list_ids
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = 1
        self.shuffle = shuffle
        self.noise_factor = noise_factor
        self.on_epoch_end()

    def __len__(self):
        """ Defines the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_ids_temp = [self.list_ids[_] for _ in indexes]
        # list_ids_temp = [_ for _ in indexes]

        X, y = self.__batch_generation(list_ids_temp)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        # self.indexes = self.list_ids
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __batch_generation(self, list_ids_temp):
        """ This method generates a batch for a denoising autoencoder
        """
        X_in = []
        for i in list_ids_temp:
            X_in.append(np.array(self.file_data[i]))
        # X_out are clean spectrograms
        X_clean = np.vstack(X_in)
        # X_in are noisy spectrograms
        X_noise = X_clean + self.noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_clean.shape)

        return X_noise, X_clean


class BatchGeneratorClass(Sequence):
    """ Generates data for Keras
    """
    def __init__(self,
                 file_data,
                 list_ids,
                 path_to_anno,
                 batch_size=batch_size_tl,
                 dim=(1, num_bands, duration_segment),
                 n_channels=1,
                 shuffle=True,
                 noise_factor=noise_factor,
                 bal_flag=True,
                 label_sel='quads'):
        """Constructor method
        """
        self.file_data = file_data
        self.list_ids = list_ids
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = 1
        self.shuffle = shuffle
        self.noise_factor = noise_factor
        self.bal_flag = bal_flag
        # load data
        annotations = pd.read_csv(path_to_anno, usecols=['Songs', 'Quads', 'Arousal', 'Valence'])
        if label_sel == 'quads':
            self.anno_list = annotations.Quads.tolist()
            self.nb_classes = 4
        elif label_sel == 'arousal':
            self.anno_list = annotations.Arousal.tolist()
            self.nb_classes = 2
        elif label_sel == 'valence':
            self.anno_list = annotations.Valence.tolist()
            self.nb_classes = 2
        # instantiate label encoder
        self.le = LabelEncoder()
        self.le.fit(self.anno_list)
        self.on_epoch_end()

    def __len__(self):
        """ Defines the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_ids_temp = [self.list_ids[_] for _ in indexes]
        # list_ids_temp = [_ for _ in indexes]

        X, y = self.batch_generation(list_ids_temp)
        # X, y, X_new, y_new = self.batch_generation(list_ids_temp)

        # one hot encode
        y_one_hot = to_categorical(self.le.transform(y), num_classes=self.nb_classes)
        # y_one_hot_new = to_categorical(self.le.transform(y_new), num_classes=nb_classes)
        # print(X.shape)
        # print(y_one_hot.shape)
        return X, y_one_hot

    def get_class_weights(self):
        y_int = self.le.fit_transform(self.anno_list)  
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_int), y_int)
        class_dict = {key: value for (key, value) in enumerate(class_weights)}
        print(class_dict)
        return class_dict

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        # self.indexes = self.list_ids
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def batch_generation(self, list_ids_temp):
        """ This method generates a batch for a denoising autoencoder
        """
        X = []
        y = []
        # print(list_ids_temp)
        for i in list_ids_temp:
            this_spec = np.array(self.file_data[i])
            X.append(this_spec)
            num_reps = this_spec.shape[0]
            # print(num_reps)
            y.append(np.repeat(self.anno_list[int(i)], num_reps).tolist())
        X = np.vstack(X)
        y = [_ for x in y for _ in x]

        if self.bal_flag and np.unique(y).shape[0] > 1:
            # test for balancing 

            # print('original X:', X.shape)
            # print('original y:', Counter(y))

            num_samp = X.shape[0]
            num_chan = X.shape[1]
            num_mel = X.shape[2]
            num_time = X.shape[3]
            X_tmp = np.reshape(X, (num_samp, num_chan * num_mel * num_time), order='C')

            sm = SMOTE(random_state=1987)
            # sm = KMeansSMOTE(random_state=1987,
            #     # kmeans_estimator=10,
            #     cluster_balance_threshold=0.1,
            #     n_jobs=-1)
            X_new, y_new = sm.fit_resample(X_tmp, y)

            X = np.reshape(X_new, (X_new.shape[0], num_chan, num_mel, num_time), order='C')
            y = y_new
            # print('Oversampled X:', X.shape)
            # print('Oversampled y:', Counter(y))

        return X, y


class BatchGeneratorCPC(Sequence):
    """ Generates data for Keras
    """
    def __init__(self,
                 file_data,
                 list_ids,
                 terms,
                 positive_samples,
                 predict_terms,
                 batch_size=batch_size_pre,
                 dim=(1, num_bands, duration_segment),
                 n_channels=1,
                 shuffle=True,
                 noise_factor=noise_factor):
        """Constructor method
        """
        self.file_data = file_data
        self.list_ids = list_ids
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = 1
        self.shuffle = shuffle
        self.noise_factor = noise_factor
        self.terms = terms
        self.positive_samples = positive_samples
        self.predict_terms = predict_terms
        self.batchy = terms + predict_terms
        self.on_epoch_end()

    def __len__(self):
        """ Defines the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_ids_temp = [self.list_ids[_] for _ in indexes]
        # list_ids_temp = [_ for _ in indexes]

        X_terms, X_pred, y = self.batch_generation(list_ids_temp)

        return [X_terms, X_pred], y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        # self.indexes = self.list_ids
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def batch_generation(self, list_ids_temp):
        """ This method generates a batch for cpc
        """
        X_terms = []
        X_pred = []
        y = []
        for i in list_ids_temp:
            this_file = np.array(self.file_data[i])
            if this_file.shape[0] >= self.batchy:
                idx_jump = np.arange(0, this_file.shape[0], self.batchy)[: this_file.shape[0] // self.batchy]
                rand_idx = np.random.permutation(idx_jump)
                idx_pos = rand_idx[:this_file.shape[0] // self.batchy // 2]
                idx_neg = rand_idx[this_file.shape[0] // self.batchy // 2:]
                for ind in idx_jump:
                    # get first terms
                    X_terms.append(this_file[ind: ind + self.terms, ...])
                    if ind in idx_pos:
                        # if positive, append following samples
                        X_pred.append(this_file[ind + self.terms: ind + self.batchy, ...])
                        y.append(1)
                    elif ind in idx_neg:
                        ord_pred = this_file[ind + self.terms: ind + self.batchy, ...]
                        rng = np.random.default_rng()
                        unord_pred = rng.permutation(ord_pred, axis=0)
                        X_pred.append(unord_pred)
                        y.append(0)
            else:
                # if file is too short, make negative example
                X_terms.append(this_file[:self.terms, ...])
                X_pred.append(this_file[:self.predict_terms, ...])
                y.append(0)

            # import matplotlib.pyplot as plt
            # print(y)
            # plt.subplot(2,6,1)
            # plt.imshow(X_terms[0][0,0,:,:], aspect='auto', origin='lower')
            # plt.subplot(2,6,2)
            # plt.imshow(X_terms[0][1,0,:,:], aspect='auto', origin='lower')
            # plt.subplot(2,6,3)
            # plt.imshow(X_terms[0][2,0,:,:], aspect='auto', origin='lower')
            # plt.subplot(2,6,4)
            # plt.imshow(X_terms[0][3,0,:,:], aspect='auto', origin='lower')
            # plt.subplot(2,6,5)
            # plt.imshow(X_pred[0][0,0,:,:], aspect='auto', origin='lower')
            # plt.subplot(2,6,6)
            # plt.imshow(X_pred[0][1,0,:,:], aspect='auto', origin='lower')


            # plt.subplot(2,6,7)
            # plt.imshow(X_terms[1][0,0,:,:], aspect='auto', origin='lower')
            # plt.subplot(2,6,8)
            # plt.imshow(X_terms[1][1,0,:,:], aspect='auto', origin='lower')
            # plt.subplot(2,6,9)
            # plt.imshow(X_terms[1][2,0,:,:], aspect='auto', origin='lower')
            # plt.subplot(2,6,10)
            # plt.imshow(X_terms[1][3,0,:,:], aspect='auto', origin='lower')
            # plt.subplot(2,6,11)
            # plt.imshow(X_pred[1][0,0,:,:], aspect='auto', origin='lower')
            # plt.subplot(2,6,12)
            # plt.imshow(X_pred[1][1,0,:,:], aspect='auto', origin='lower')
 
            # plt.show()
            # pdb.set_trace()

        # print(y)
        try:
            for i, _ in enumerate(X_terms):
                # print(_.shape)
                if _.shape[0] != self.terms:
                    X_terms.pop(i)
                    X_pred.pop(i)
                    y.pop(i)
                    print('popped term', i)
            for i, _ in enumerate(X_pred):
                # print(_.shape)
                if _.shape[0] != self.predict_terms:
                    X_terms.pop(i)
                    X_pred.pop(i)
                    y.pop(i)
                    print('popped prediction', i)
            X_terms = np.stack(X_terms)
            X_pred = np.stack(X_pred)
            # pdb.set_trace()
        except:
            print(len(X_pred))
            print(len(X_terms))
            print(y)
            X_terms = np.expand_dims(X_terms[0], axis=0)
            X_pred = np.expand_dims(X_pred[0], axis=0)
            y = [y[0]]


        return X_terms, X_pred, y


class BatchGeneratorClassMulti(Sequence):
    """ Generates data for Keras
    """
    def __init__(self,
                 file_data,
                 list_ids,
                 path_to_anno,
                 batch_size=batch_size_tl,
                 dim=(1, num_bands, duration_segment),
                 n_channels=1,
                 shuffle=True,
                 noise_factor=noise_factor,
                 bal_flag=True,
                 label_sel='quads'):
        """Constructor method
        """
        self.file_data = file_data
        self.list_ids = list_ids
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = 1
        self.shuffle = shuffle
        self.noise_factor = noise_factor
        self.bal_flag = bal_flag
        # load data
        annotations = pd.read_csv(path_to_anno, usecols=['Songs', 'Quads', 'Arousal', 'Valence'])

        self.anno_list_quads = annotations.Quads.tolist()
        self.anno_list_arousal = annotations.Arousal.tolist()
        self.anno_list_valence = annotations.Valence.tolist()

        # instantiate label encoder
        self.le1 = LabelEncoder()
        self.le2 = LabelEncoder()
        self.le3 = LabelEncoder()
        self.le1.fit(self.anno_list_quads)
        self.le2.fit(self.anno_list_arousal)
        self.le3.fit(self.anno_list_valence)
        self.on_epoch_end()

    def __len__(self):
        """ Defines the number of batches per epoch"""
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        list_ids_temp = [self.list_ids[_] for _ in indexes]
        # list_ids_temp = [_ for _ in indexes]

        X, y1, y2, y3 = self.batch_generation(list_ids_temp)

        # one hot encode
        y1_enc = to_categorical(self.le1.transform(y1), num_classes=4)
        y2_enc = to_categorical(self.le2.transform(y2), num_classes=2)
        y3_enc = to_categorical(self.le3.transform(y3), num_classes=2)
        # y2_enc = self.le2.transform(y2)
        # y3_enc = self.le3.transform(y3)

        return X, [y1_enc, y2_enc, y3_enc]

    def get_class_weights(self):
        y_int = self.le.fit_transform(self.anno_list)  
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_int), y_int)
        class_dict = {key: value for (key, value) in enumerate(class_weights)}
        print(class_dict)
        return class_dict

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_ids))
        # self.indexes = self.list_ids
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def batch_generation(self, list_ids_temp):
        """ This method generates a batch for a denoising autoencoder
        """
        X = []
        y1 = []
        y2 = []
        y3 = []
        # print(list_ids_temp)
        for i in list_ids_temp:
            this_spec = np.array(self.file_data[i])
            X.append(this_spec)
            num_reps = this_spec.shape[0]
            # print(num_reps)
            y1.append(np.repeat(self.anno_list_quads[int(i)], num_reps).tolist())
            y2.append(np.repeat(self.anno_list_arousal[int(i)], num_reps).tolist())
            y3.append(np.repeat(self.anno_list_valence[int(i)], num_reps).tolist())
        X = np.vstack(X)
        y1 = [_ for x in y1 for _ in x]
        y2 = [_ for x in y2 for _ in x]
        y3 = [_ for x in y3 for _ in x]
        return X, y1, y2, y3


if __name__ == '__main__':
    # following just for tests
    # usage python3 batch_generator.py -d s -l e/m
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset',
                        help='Select to generate data for music (m) or speech (s)',
                        action='store',
                        required=True,
                        dest='dataset')
    parser.add_argument('-l',
                        '--language',
                        help='Select language of data: english (e) or mandarin (m)',
                        action='store',
                        required=True,
                        dest='language')
    args = parser.parse_args()

    if args.dataset == 'm' and args.language == 'e':
        path_to_data = path_music_feat_eng
        path_to_anno = path_music_anno_eng
        print('Path to data:', path_to_data)
    elif args.dataset == 'm' and args.language == 'm':
        path_to_data = path_music_feat_man
        path_to_anno = path_music_anno_man
        print('Path to data:', path_to_data) 
    elif args.dataset == 's' and args.language == 'e':
        path_to_data = path_speech_feat_eng
        path_to_anno = path_speech_anno_eng
        print('Path to data:', path_to_data)
    elif args.dataset == 's' and args.language == 'm':
        path_to_data = path_speech_feat_man
        path_to_anno = path_speech_anno_man
        print('Path to data:', path_to_data)

    # CASE BATCH GENERATOR FOR AUTOENCODER
    TRAIN_TENSOR = os.path.join(path_to_data, 'train_dataset.h5py')
    train_tensor = h5py.File(TRAIN_TENSOR, 'r')
    train_keys = [_ for _ in train_tensor.keys()]
    # generator = BatchGeneratorAutoenc(train_tensor, train_keys)

    # CASE BATCH GENERATOR FOR CLASSIFIER
    # generator = BatchGeneratorClass(train_tensor, train_keys, path_to_anno, label_sel='arousal')
    # X, y = generator.__getitem__(0)

    # # CASE BATCH GENERATOR FOR CPC
    # generator = BatchGeneratorCPC(train_tensor, train_keys, terms=terms, positive_samples=positive_samples, predict_terms=predict_terms)

    # [X_terms, X_pred], y = generator.__getitem__(0)

    # CASE BATCH GENERATOR FOR Multi-task CLASSIFIER
    generator = BatchGeneratorClassMulti(train_tensor, train_keys, path_to_anno)
    X, y = generator.__getitem__(0)

    pdb.set_trace()
    # # testing for noisy labels
    # # from cleanlab.classification import LearningWithNoisyLabels
    # from cleanlab.noise_generation import generate_noise_matrix_from_trace
    # from cleanlab.noise_generation import generate_noisy_labels
    # import models.model_cnn
    # from sklearn.metrics import classification_report


    # generator = BatchGeneratorClass(train_tensor, train_keys, path_to_anno, batch_size=200)
    # X, y = generator.__getitem__(0)

    # py = np.bincount(np.argmax(y, axis=1))/ float(y.shape[1])
    # noise_matrix = generate_noise_matrix_from_trace(K=nb_classes,
    #                                                trace=nb_classes*0.65,
    #                                                py=py,
    #                                                frac_zero_noise_rates=0.0)


    # np.random.seed(seed=1)
    # y_int_err = generate_noisy_labels(np.argmax(y, axis=1), noise_matrix)
    # y_noise = generator.le.fit_transform(y_int_err)
    # y_noise_cat = to_categorical(y_noise)

    # model_cle = models.model_cnn.build_model(lrate=1e-4, decay=1e-6)
    # model_noi = models.model_cnn.build_model(lrate=1e-4, decay=1e-6)

    # c_w = generator.get_class_weights()

    # hist_clean = model_cle.fit(X, y, class_weight=c_w, epochs=5)

    # hist_noise = model_noi.fit(X, y_noise_cat, class_weight=c_w, epochs=5)

    # y_true = np.argmax(y, axis=1)
    # y_pred_cle = model_cle.predict(X)
    # y_pred_noi = model_noi.predict(X)

    # y_pred_cle_int = np.argmax(y_pred_cle, axis=1)
    # y_pred_noi_int = np.argmax(y_pred_noi, axis=1)

    # print(classification_report(y_true, y_pred_cle_int))
    # print(classification_report(y_true, y_pred_noi_int))

    # pdb.set_trace()
