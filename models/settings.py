import numpy as np

# use source separation [True] or not [False]
unmix = False
if unmix:
    unmix_txt = '_sep'
else:
    unmix_txt = ''
# work with spectrogram [True] or chroma [False]
feats_melspec = True
# work with primewords [True] or aishell [False]
primewords = True
# work with ntwicm [True] or 4q_emotions [False]
ntwicm = False
# sampling rate
sr_low = True
if sr_low:
    sr_txt = '_16k'
    time_frames = 31
else:
    sr_txt = ''
    time_frames = 43
# training params
nb_epochs_pre = 100
nb_epochs_tl1 = 300
nb_epochs_tl2 = 400

# Model parameters
time_duration = 1  # duration of mel spectrogram patches (seconds)
# time frames for spectrogram patches
duration_segment = int(time_frames * time_duration)
test_overlap = 0.5  # spectrograms overlapping for test data set generation
# feature extraction
num_bands = 128  # mel bands in melspectrogram
nb_classes = 4  # number of classes for the classifier
val_split = 0.85  # train and validation split
# denoiser autoencoder
noise_factor = 0.3  #
subset_speech = True  # flag to subset the amount of pre-training data set
subset_rate = 0.15  # percentage of speech data set to use

# training parameters
batch_size_pre = 8  # mini batch size (Han2016)
batch_size_tl = 40
learning_rate_pre = 0.001  #optimizer learning rate
learning_rate_pre = 0.1
learning_rate_tl1 = 0.0001
learning_rate_tl2 = 0.00005
alpha = 0.33  # alpha for leakyReLU activation function (Han2016)
# minimum float32 representation epsilon in python
eps = np.finfo(np.float32).eps
# contrastive predictive coding parameters
terms = 2
positive_samples = 2
predict_terms = 4



home = True

if not home:
    print('*****\n*****\nWorking at UPF\n*****\n*****')
    # english
    path_speech_eng = '/media/juangomez/DADES/datasets/speech/librispeech/LibriSpeech_WAV'
    path_speech_anno_eng = '/media/juangomez/DADES/datasets/speech/librispeech/LibriSpeech_WAV/out.csv'        
    if ntwicm:
        path_music_eng = '/media/juangomez/DADES/datasets/emotions/NTWICM/Now.Thats.What.I.Call.Music_Complete.Collection-(1-75)-TPB/NTWICM_mp3'
        path_music_anno_eng = '/media/juangomez/DADES/datasets/emotions/NTWICM/Now.Thats.What.I.Call.Music_Complete.Collection-(1-75)-TPB/NTWICM_mp3/out.csv'
    else:
        path_music_eng = '/media/juangomez/DADES/datasets/emotions/4Q_emotions/4Q_emotions_mp3'
        path_music_anno_eng = '/media/juangomez/DADES/datasets/emotions/4Q_emotions/4Q_emotions_mp3/4Q_Quads.csv'
    # mandarin
    path_music_man = '/media/juangomez/DADES/datasets/emotions/CH818/ch818_mp3'
    path_music_anno_man = '/media/juangomez/DADES/datasets/emotions/CH818/ch818_mp3/CH818_Quads.csv'
    if feats_melspec:
        path_speech_feat_eng = '/media/juangomez/DADES/datasets/speech/librispeech/LibriSpeech_specs{}'.format(sr_txt)
        if ntwicm:
            path_music_feat_eng = '/media/juangomez/DADES/datasets/emotions/NTWICM/Now.Thats.What.I.Call.Music_Complete.Collection-(1-75)-TPB/NTWICM_specs{}'.format(sr_txt)
        elif not ntwicm:    
            path_music_feat_eng = '/media/juangomez/DADES/datasets/emotions/4Q_emotions/4Q_emotions_specs{}'.format(sr_txt)
        if primewords:
            path_speech_man = '/media/juangomez/DADES/datasets/speech/primewords/primewords_2018'
            path_speech_anno_man = '/media/juangomez/DADES/datasets/speech/primewords/primewords_2018/out.csv'
            path_speech_feat_man = '/media/juangomez/DADES/datasets/speech/primewords/primewords_specs{}'.format(sr_txt)
        else:
            path_speech_man = '/media/juangomez/DADES/datasets/speech/aishell/aishell_2017'
            path_speech_anno_man = '/media/juangomez/DADES/datasets/speech/aishell/aishell_2017/out.csv'
            path_speech_feat_man = '/media/juangomez/DADES/datasets/speech/aishell/aishell_specs{}'.format(sr_txt)
        path_music_feat_man = '/media/juangomez/DADES/datasets/emotions/CH818/ch818_specs{}'.format(sr_txt)

    else:
        path_speech_feat_eng = '/media/juangomez/DADES/datasets/speech/librispeech/LibriSpeech_chroma'
        path_music_feat_eng = '/media/juangomez/DADES/datasets/emotions/4Q_emotions/4Q_emotions_chroma'
        if primewords:
            path_speech_man = '/media/juangomez/DADES/datasets/speech/primewords/primewords_2018'
            path_speech_anno_man = '/media/juangomez/DADES/datasets/speech/primewords/primewords_2018/out.csv'
            path_speech_feat_man = '/media/juangomez/DADES/datasets/speech/primewords/primewords_chroma'
        else:
            path_speech_man = '/media/juangomez/DADES/datasets/speech/aishell/aishell_2017'
            path_speech_anno_man = '/media/juangomez/DADES/datasets/speech/aishell/aishell_2017/out.csv'
            path_speech_feat_man = '/media/juangomez/DADES/datasets/speech/aishell/aishell_chroma'
        path_music_feat_man = '/media/juangomez/DADES/datasets/emotions/CH818/ch818_chroma'

else:
    print('*****\n*****\nWorking at home\n*****\n*****')
    # english
    path_speech_eng = '/media/hoodoochild/DATA/datasets/speech/librispeech/LibriSpeech_wav'
    path_speech_anno_eng = '/media/hoodoochild/DATA/datasets/speech/librispeech/LibriSpeech_wav/out.csv'
    if ntwicm:
        path_music_eng = '/media/hoodoochild/DATA/datasets/NTWICM/NTWICM_mp3{}'.format(unmix_txt)
        path_music_anno_eng = '/media/hoodoochild/DATA/datasets/NTWICM/NTWICM_mp3/out.csv'
    else:
        path_music_eng = '/media/hoodoochild/DATA/datasets/MER_audio_taffc_dataset/4Q_emotions_mp3{}'.format(unmix_txt)
        path_music_anno_eng = '/media/hoodoochild/DATA/datasets/MER_audio_taffc_dataset/4Q_emotions_mp3/4Q_Quads.csv'
    # mandarin
    path_music_man = '/media/hoodoochild/DATA/datasets/CH818/ch818_mp3{}'.format(unmix_txt)
    path_music_anno_man = '/media/hoodoochild/DATA/datasets/CH818/ch818_mp3/CH818_Quads.csv'
    if feats_melspec:
        path_speech_feat_eng = '/media/hoodoochild/DATA/datasets/speech/librispeech/LibriSpeech_specs{}'.format(sr_txt)
        if ntwicm:
            path_music_feat_eng = '/media/hoodoochild/DATA/datasets/NTWICM/NTWICM_specs{}'.format(sr_txt)
        else:
            path_music_feat_eng = '/media/hoodoochild/DATA/datasets/MER_audio_taffc_dataset/4Q_emotions_specs{}{}'.format(sr_txt, unmix_txt)
        if primewords:
            path_speech_man = '/media/hoodoochild/DATA/datasets/speech/primewords/primewords_2018'
            path_speech_anno_man = '/media/hoodoochild/DATA/datasets/speech/primewords/primewords_2018/out.csv'    
            path_speech_feat_man = '/media/hoodoochild/DATA/datasets/speech/primewords/primewords_specs{}'.format(sr_txt)
        else:
            path_speech_man = '/media/hoodoochild/DATA/datasets/speech/aishell/aishell_2017'
            path_speech_anno_man = '/media/hoodoochild/DATA/datasets/speech/aishell/aishell_2017/out.csv'    
            path_speech_feat_man = '/media/hoodoochild/DATA/datasets/speech/aishell/aishell_specs{}'.format(sr_txt)
        path_music_feat_man = '/media/hoodoochild/DATA/datasets/CH818/ch818_specs{}{}'.format(sr_txt, unmix_txt)
    else:
        path_speech_feat_eng = '/media/hoodoochild/DATA/datasets/speech/librispeech/LibriSpeech_chroma'
        path_music_feat_eng = '/media/hoodoochild/DATA/datasets/MER_audio_taffc_dataset/4Q_emotions_chroma'
        if primewords:
            path_speech_man = '/media/hoodoochild/DATA/datasets/speech/primewords/primewords_2018'
            path_speech_anno_man = '/media/hoodoochild/DATA/datasets/speech/primewords/primewords_2018/out.csv'    
            path_speech_feat_man = '/media/hoodoochild/DATA/datasets/speech/primewords/primewords_chroma'
        else:
            path_speech_man = '/media/hoodoochild/DATA/datasets/speech/aishell/aishell_2017'
            path_speech_anno_man = '/media/hoodoochild/DATA/datasets/speech/aishell/aishell_2017/out.csv'    
            path_speech_feat_man = '/media/hoodoochild/DATA/datasets/speech/aishell/aishell_chroma'
        path_music_feat_man = '/media/hoodoochild/DATA/datasets/CH818/ch818_chroma'

