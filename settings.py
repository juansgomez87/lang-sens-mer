import numpy as np

# use source separation [True] or not [False]
unmix = False
if unmix:
    unmix_txt = '_sep'
else:
    unmix_txt = ''

# sampling rate
sr_low = True
if sr_low:
    sr_txt = '_16k'
    time_frames = 31
    sampling_rate = 16000
else:
    sr_txt = ''
    time_frames = 43
    sampling_rate = 22050
# training params
nb_epochs_pre = 100
nb_epochs_tl1 = 100
nb_epochs_tl2 = 100

# Model parameters
time_duration = 1  # duration of mel spectrogram patches (seconds)
# time frames for spectrogram patches
duration_segment = int(time_frames * time_duration)
test_overlap = 0.5  # spectrograms overlapping for test data set generation
# feature extractions
num_bands = 128  # mel bands in melspectrogram
nb_classes = 4  # number of classes for the classifier
val_split = 0.85  # train and validation split
# denoiser autoencoder
noise_factor = 0.3  #
feats_melspec = True

subset_speech = True  # flag to subset the amount of pre-training data set
subset_rate = 0.075 # percentage of speech data set to use aishel 0.043, librispeech 0.075

# training parameters
batch_size_pre = 4  # mini batch size (Han2016)
batch_size_tl = 4
# pre training denoising
# librispeech:
# acc 0.624344 - lr 0.0009653265915661777 decay 4.244417355110645e-06
# aishell
# acc 0.63873 - lr 0.0017 decay 4.0810240705777585e-06
# pretraining cpc
# acc 0.89 - lr 0.001 decay  0.0038
learning_rate_pre = 0.001  #optimizer learning rate
decay_pre = 4e-6

learning_rate_tl1 = 0.001
learning_rate_tl2 = 0.00005
decay_tl = 0.0001
alpha = 0.33  # alpha for leakyReLU activation function (Han2016)
# minimum float32 representation epsilon in python
eps = np.finfo(np.float32).eps
# contrastive predictive coding parameters
positive_samples = 2
predict_terms = 2
terms = 4

path_to_data = ''
# english
path_speech_eng = f'{path_to_data}/speech/librispeech/LibriSpeech_wav'
path_speech_anno_eng = f'{path_to_data}/speech/librispeech/LibriSpeech_wav/out_30h.csv'
path_speech_feat_eng = f'{path_to_data}/speech/librispeech/LibriSpeech_specs{}'.format(sr_txt)
path_speech_is13_eng = f'{path_to_data}/speech/librispeech/LibriSpeech_is13{}'.format(sr_txt)
path_music_eng = f'{path_to_data}/MER_audio_taffc_dataset/4Q_emotions_mp3_3h{}'.format(unmix_txt)
path_music_anno_eng = f'{path_to_data}/MER_audio_taffc_dataset/4Q_emotions_mp3_3h/4Q_Quads_3h.csv'
path_music_feat_eng = f'{path_to_data}/MER_audio_taffc_dataset/4Q_emotions_specs_3h{}{}'.format(sr_txt, unmix_txt)
path_music_is13_eng = f'{path_to_data}/MER_audio_taffc_dataset/4Q_emotions_is13_3h{}{}'.format(sr_txt, unmix_txt)
# mandarin
path_speech_man = f'{path_to_data}/speech/aishell/aishell_2017'
path_speech_anno_man = f'{path_to_data}/speech/aishell/aishell_2017/out_30h.csv'    
path_speech_feat_man = f'{path_to_data}/speech/aishell/aishell_specs{}'.format(sr_txt)
path_speech_is13_man = f'{path_to_data}/speech/aishell/aishell_is13{}'.format(sr_txt)
path_music_man = f'{path_to_data}/CH818/ch818_mp3{}'.format(unmix_txt)
path_music_anno_man = f'{path_to_data}/CH818/ch818_mp3/CH818_Quads.csv'
path_music_feat_man = f'{path_to_data}/CH818/ch818_specs{}{}'.format(sr_txt, unmix_txt)
path_music_is13_man = f'{path_to_data}/CH818/ch818_is13{}{}'.format(sr_txt, unmix_txt)
# mix
path_speech_mix = f'{path_to_data}/speech/mix_man_eng_30h/audio'
path_speech_anno_mix = f'{path_to_data}/speech/mix_man_eng_30h/audio/out_30h.csv'
path_speech_feat_mix = f'{path_to_data}/speech/mix_man_eng_30h/specs{}'.format(sr_txt)
path_speech_is13_mix = f'{path_to_data}/speech/mix_man_eng_30h/is13{}'.format(sr_txt)
