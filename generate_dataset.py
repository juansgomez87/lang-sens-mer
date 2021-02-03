import librosa
import pandas as pd
import numpy as np
import argparse
import os
import shutil
import time
import pdb
import h5py
import multiprocessing as mp
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from settings import *
import click


class GenerateDataset:
    """ GenerateDataset generates training and testing tensors for a given dataset.
    All of the parameters are inherited from *settings.py*.

    :param duration: number of time bins to create melspectrogram.
    :param num_classes: number of classes to be evaluated.
    :param val_split: training-validation split from 0 to 1.
    :type duration: int
    :type num_classes: int
    :type val_split: float
    """
    def __init__(self,
                 path_to_data,
                 path_to_save,
                 path_anno,
                 duration,
                 num_classes,
                 val_split):
        """ Constructor method
        """
        # define overlap for feature extraction
        # TODO define ovl from settings?
        self.long_fids = mp.Manager().list()
        self.long_len = mp.Manager().list()
        ovl = True
        if not ovl:
            print('Extracting features with no overlap')
            self.chunk = duration
        else:
            print('Extracting features with {} overlap'.format(test_overlap))
            self.chunk = int(duration * test_overlap)

        self.list_quad = ['Q1', 'Q2', 'Q3', 'Q4']
        self.duration = duration  # set segment duration in settings file
        self.num_classes = num_classes  # set number of classes in settings file
        annotations = pd.read_csv(path_anno, usecols=['Songs', 'Quads'])
        all_fid = annotations.index.tolist()

        # copy dataset structure folders
        self.copy_folder(path_to_data, path_to_save)
        
        # save class weights
        anno_list = annotations.Quads.tolist()
        le = LabelEncoder()
        y_int = le.fit_transform(anno_list)
        class_weights = class_weight.compute_class_weight('balanced', np.unique(y_int), y_int)
        class_dict = {key: value for (key, value) in enumerate(class_weights)}

        np.save(os.path.join(path_to_save, 'class_weights.npy'), class_dict)

        # Data sets file paths
        TRAIN_TENSOR = os.path.join(path_to_save, 'train_dataset.h5py')
        train_tensor = h5py.File(TRAIN_TENSOR, 'w')
        VAL_TENSOR = os.path.join(path_to_save, 'val_dataset.h5py')
        val_tensor = h5py.File(VAL_TENSOR, 'w')
        if self.num_classes is not None:
            TEST_TENSOR = os.path.join(path_to_save, 'test_dataset.h5py')
            test_tensor = h5py.File(TEST_TENSOR, 'w')

        # balance data sets by undersampling
        undersampling = False
        fids_per_class = {}
        for quad in self.list_quad:
            fids_per_class[quad] = np.where(annotations.Quads == quad)[0].tolist()
        min_num_class = np.min([len(fids_per_class[_]) for _ in self.list_quad])
        max_num_class = np.max([len(fids_per_class[_]) for _ in self.list_quad])
        if min_num_class != max_num_class:
            print('Undersampling to less represented class {} with {} files.'.format(np.argmin(fids_per_class), min_num_class))
            undersampling = True

        undersampling = False
        # pdb.set_trace()
        if self.num_classes is not None:
            if undersampling:
                # random split 50/50 for training and testing
                train_full_fid, test_fid = self.data_split(annotations, 0.7, n_f=min_num_class)
            else:
                train_full_fid, test_fid = self.data_split(annotations, 0.7, n_f=None)
            # random split for training and validation data set
            train_fid, val_fid = self.data_split(annotations.loc[train_full_fid], val_split, n_f=None)

        else:
            # annotations = annotations.sample(frac=subset_rate)
            all_fid = annotations.index.tolist()
            # all_fid_short = annotations_short.index.tolist()
            # random split for training and validation data set
            # train_fid, val_fid = self.data_split(annotations_short, val_split, n_f=None)


        # select all files and file names
        in_files = annotations.Songs.tolist()
        self.format_data = in_files[0].split('.')[-1]
        # if unmix flag, select vocals
        if unmix:
            in_files = [_.replace('.mp3', '/vocals.wav') for _ in in_files]
        form_data = in_files[0].split('.')[-1]
        out_files = [_.replace('./', path_to_save +'/').replace(form_data, 'npy') for _ in in_files]

        # generate output filenames
        in_files = [_.replace('./', path_to_data+'/') for _ in in_files]
        zip_files = [(in_files[_], all_fid[_], out_files[_]) for _ in range(len(in_files))]

        # extract all data!
        print('Extracting complete data set...')
        start = time.time()
        pool = mp.Pool(processes=mp.cpu_count())

        print('Mel-spectrograms...')
        # calculate the melspectrograms for all files in the file input list
        pool.starmap(self.create_spectrogram, zip_files)
        pool.close()
        pool.join()
        print('Duration:', (time.time() - start) / 60, ' minutes.')

        debugging = False
        pdb.set_trace()
        if debugging:
            self.long_fids = [_ for _ in self.long_fids]
            self.long_len = [_ for _ in self.long_len]
            print('{} hours of data'.format(np.sum(self.long_len)/60/60))

            cnt = 0
            cum_len = 0
            for i, length in enumerate(self.long_len):
                cum_len += length
                cnt += 1
                # if cum_len >= 24600:  # equivalent to 30s * 2 specs per sec * 60s * 60 min = 3 hrs
                # #     break
                # if cum_len >= 49200:  # equivalent to 30s * 2 specs per sec * 60s * 60 min = 6 hrs
                #     break
                # if cum_len >= 73800:  # equivalent to 30s * 2 specs per sec * 60s * 60 min = 12 hrs
                #     break
                # if cum_len >= 98400:  # equivalent to 30s * 2 specs per sec * 60s * 60 min = 15 hrs
                #     break
                if cum_len >= 246000:  # equivalent to 30s * 2 specs per sec * 60s * 60 min = 30 hrs
                    break

            # pdb.set_trace()
            annotations = annotations.iloc[self.long_fids[:cnt]]
            annotations['orig_fid'] = self.long_fids[:cnt]
            annotations['len_sec'] = self.long_len[:cnt]
            pdb.set_trace()
            annotations.to_csv('out_30h.csv')

            train_fid, val_fid = self.data_split(annotations, val_split, n_f=None)
        else:
            train_fid, val_fid = self.data_split(annotations, val_split, n_f=None)


        print('Exporting train data set...')
        self.export_data(annotations, train_fid, train_tensor, path_to_save)
        print('Duration:', (time.time() - start) / 60, ' minutes.')

        # extract each validation data set
        print('Extracting validation data set...')
        start = time.time()
        self.export_data(annotations, val_fid, val_tensor, path_to_save)
        print('Duration:', (time.time() - start) / 60, ' minutes.')

        if self.num_classes is not None:
            # extract each pure test data set
            print('Extracting pure test data set...')
            start = time.time()
            self.export_data(annotations, test_fid, test_tensor, path_to_save)
            print('Duration:', (time.time() - start) / 60, ' minutes.')


    def data_split(self, df, val_split, n_f):
        """
        """
        split1_fid = []
        split2_fid = []
        if self.num_classes is not None:
            for quad in self.list_quad:
                # randomize rows for each quadrant
                if n_f is not None:
                    rand_Q_df = df[df.Quads == quad].sample(n_f)
                elif n_f is None:
                    rand_Q_df = df[df.Quads == quad].sample(frac=1)
                # number of indexes
                n_idx = len(rand_Q_df.index)
                split1_fid.extend([_ for _ in rand_Q_df.index][:int(n_idx * val_split)])
                split2_fid.extend([_ for _ in rand_Q_df.index][int(n_idx * val_split):])
        else:
            rand_df = df.sample(frac=1)
            n_idx = len(rand_df.index)
            split1_fid.extend([_ for _ in rand_df.index][:int(n_idx * val_split)])
            split2_fid.extend([_ for _ in rand_df.index][int(n_idx * val_split):])
        return split1_fid, split2_fid

    def create_spectrogram(self, in_f, fid, out_f, sr=sampling_rate, win_length=1024, hop_length=512, num_mel=num_bands):
        """This method creates a melspectrogram from an audio file using librosa
        audio processing library. Parameters are default from Han et al.
        
        :param filename: wav filename to process.
        :param sr: sampling rate in Hz (default: 22050).
        :param win_length: window length for STFT (default: 1024).
        :param hop_length: hop length for STFT (default: 512).
        :param num_mel: number of mel bands (default:128).
        :type filename: str
        :type sr: int
        :type win_length: int
        :type hop_length: int
        :type num_mel: int
        :returns: **ln_S** *(np.array)* - melspectrogram of the complete audio file with logarithmic compression with dimensionality [mel bands x time frames].
        """
        assert os.path.exists(in_f), "filename %r does not exist" % in_f
        if os.path.exists(out_f):
            print('File {} already created, jumping to next...'.format(fid))
            if self.num_classes is None:
                try:
                    test = np.load(out_f)
                    if test.shape[0] > 6:
                        print('long file appended!')
                        self.long_fids.append(fid)
                        self.long_len.append(test.shape[0])
                except:
                    print('prob file!')
        else:
            data, sr = librosa.load(in_f, sr=sr, mono=True)
            try:
                # normalize data
                data /= np.max(np.abs(data))
            except Warning:
                print(filename, ' is empty')

            # time-frequency representation Short Time Fourier Transform
            D = np.abs(librosa.stft(data, win_length=win_length, hop_length=hop_length, center=True))
            # mel frequency representation
            S = librosa.feature.melspectrogram(S=D, sr=sr, n_mels=num_mel)
            # apply natural logarithm
            ln_S = np.log(S + eps)
            spec_list = []
            for idx in range(0, ln_S.shape[1] - self.duration + 1, self.chunk):
                # append chunk of spectrogram to dataset
                spec_list.append(ln_S[:, idx:(idx + self.duration)])
            spec_array = np.expand_dims(spec_list, axis=1)
            if spec_array.shape[0] > 8:
                print('long file appended!')
                self.long_fids.append(fid)
                self.long_len.append(spec_array.shape[0])
            np.save(out_f, spec_array)

            print('File {} has been saved!'.format(fid))



    def export_data(self, d_f, fid, tensor_f, path_to_save):
        """ This method loads all selected files and saves the corresponding 
        data set"""
        in_files = d_f.Songs[fid].tolist()
        form_dat = '.' + self.format_data
        if unmix:
            sel_files = [_.replace('./', path_to_save+'/').replace(form_dat, '/vocals.npy') for _ in in_files]
        else:
            sel_files = [_.replace('./', path_to_save+'/').replace(self.format_data, 'npy') for _ in in_files]

        num_files = len(sel_files)
        prob_files = []
        fill_char = click.style('=', fg='blue')
        with click.progressbar(range(len(fid)), label='Loading...', fill_char=fill_char) as bar:
            for idx, f, this_fid, i in zip(range(num_files), sel_files, fid, bar):
                # print('Processing file {}/{}'.format(idx, num_files))
                try:
                    this_file = np.load(f)
                    try:
                        tensor_f.create_dataset(str(this_fid), data=this_file)
                    except RuntimeError:
                        print('Skip file {}/{}!')
                except ValueError:
                    prob_files.append(f)
        print('Problem files: ', prob_files)

    def copy_folder(self, in_folder, out_folder):
        if not(os.path.isdir(out_folder)):
            shutil.copytree(in_folder, out_folder, ignore=self.ig_f)


    def ig_f(self, dir, files):
        return [f for f in files if os.path.isfile(os.path.join(dir, f))]


if __name__ == '__main__':
    # Usage python3 generate_dataset.py --dataset m/s --language e/m
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset',
                        help='Select to generate data for music (m) or speech (s)',
                        action='store',
                        required=True,
                        dest='dataset')
    parser.add_argument('-l',
                        '--language',
                        help='Select language of data: english (e) or mandarin (m) or mix (x)',
                        action='store',
                        required=True,
                        dest='language')
    parser.add_argument('-pl',
                        '--path_load',
                        help='Select a specific path to load the data',
                        action='store',
                        dest='path_load')
    parser.add_argument('-ps',
                        '--path_save',
                        help='Select a specific path to save the data',
                        action='store',
                        dest='path_save')
    args = parser.parse_args()


    if args.dataset == 'm' and args.language == 'e':
        path_to_data = path_music_eng
        path_anno = path_music_anno_eng
        path_to_save = path_music_feat_eng
        print('Path to save:', path_to_save)
    elif args.dataset == 'm' and args.language == 'm':
        path_to_data = path_music_man
        path_anno = path_music_anno_man
        path_to_save = path_music_feat_man
        print('Path to save:', path_to_save)
    elif args.dataset == 's' and args.language == 'e':
        path_to_data = path_speech_eng
        path_anno = path_speech_anno_eng
        path_to_save = path_speech_feat_eng
        nb_classes = None
        print('Processing english speech, no classes!')
        print('Path to save:', path_to_save)
    elif args.dataset == 's' and args.language == 'm':
        path_to_data = path_speech_man
        path_anno = path_speech_anno_man
        path_to_save = path_speech_feat_man
        nb_classes = None
        print('Processing mandarin speech, no classes!')
        print('Path to save:', path_to_save)
    elif args.dataset == 's' and args.language == 'x':
        path_to_data = path_speech_mix
        path_anno = path_speech_anno_mix
        path_to_save = path_speech_feat_mix
        nb_classes = None
        print('Processing mandarin speech, no classes!')
        print('Path to save:', path_to_save)

    # time process
    start = time.time()
    mp.freeze_support()
    # initialize dataset
    data = GenerateDataset(path_to_data,
                           path_to_save,
                           path_anno,
                           duration_segment,
                           nb_classes,
                           val_split)

    print('Total processing time: ', (time.time() - start) / 60, 'minutes.')