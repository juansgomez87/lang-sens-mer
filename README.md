# Language-sensitive Music Emotion Recognition models: are we really there yet?

Our previous research showed promising results when transferring features learned from speech to train emotion recognition models for music. 
In this context, we implemented a denoising autoencoder as a pretraining approach to extract features from speech in two languages (English and Mandarin). 
From that, we performed transfer and multi-task learning to predict classes from the arousal-valence space of music emotion. 
We tested and analyzed intra-linguistic and cross-linguistic settings, depending on the language of speech and lyrics of the music. 

This paper presents additional investigation on our approach, which reveals that: (1) performing pretraining with speech in a mixture of languages yields similar results than for specific languages - the pretraining phase appears not to exploit particular language features, (2) the music in Mandarin dataset consistently results in poor classification performance - we found low agreement in annotations, and (3) novel methodologies for representation learning (Contrastive Predictive Coding) may exploit features from both languages (i.e., pretraining on a mixture of languages) and improve classification of music emotions in both languages.
From this study we conclude that more research is still needed to understand what is actually being transferred in these type of contexts. 

![CPC implementation][cpc]

[cpc]: https://github.com/juansgomez87/lang-sens-mer/blob/master/img/cpc_graph.png "CPC implementation"

## Usage
Following, we explain the steps to reproduce the the results from our paper. Note that all settings are set in the `settings.py` file. Simply change the `path_to_data` to the directory of your choice. 

Additionally, you need to download the fully trained models from [here](https://drive.google.com/file/d/12RVXvA53bQ70fRRCDc70iQZCF0EFSS4f/view?usp=sharing). Extract the directories `models` and `models_trans` to the home directory.

### Install requirements

Use:
```
pip3 install -r requirements.txt
```

### Extract features
In order to extract the datasets use the following script: 

```
python3 generate_dataset.py --dataset m/s --language e/m
```
`--dataset` is m for music or s for speech and `language` is e for english or m for mandarin.

### Pre-training
For pretraining, two scripts are available: SCAE and CPC.

To pretrain with the SCAE model and all languages use:
```
python3 pretrain_denoise.py -s e -mod model_over && python3 pretrain_denoise.py -s m -mod model_over && python3 pretrain_denoise.py -s x -mod model_over 
```

To pretrain with the CPC model and all languages use:
```
python3 pretrain_cpc.py -s e -mod model_cpc && python3 pretrain_cpc.py -s m -mod model_cpc && python3 pretrain_cpc.py -s x -mod model_cpc 
```

### Transfer learning

For transfer learning, two configurations are available: feature extractor and full. This results in four configurations: SCAE-Feat. Ext., SCAE-Full, CPC-Feat. Ext., CPC-Full.

To perform transfer learning with SCAE-Feat. Ext and all possible combinations:
```
python3 trans_train_denoise_multi.py -s e -m e -mod model_over -rel n && python3 trans_train_denoise_multi.py -s e -m m -mod model_over -rel n && python3 trans_train_denoise_multi.py -s m -m e -mod model_over -rel n && python3 trans_train_denoise_multi.py -s m -m m -mod model_over -rel n && python3 trans_train_denoise_multi.py -s x -m e -mod model_over -rel n && python3 trans_train_denoise_multi.py -s x -m m -mod model_over -rel n
```
To perform transfer learning with SCAE-Full and all possible combinations:
```
python3 trans_train_denoise_multi.py -s e -m e -mod model_over -rel y && python3 trans_train_denoise_multi.py -s e -m m -mod model_over -rel y && python3 trans_train_denoise_multi.py -s m -m e -mod model_over -rel y && python3 trans_train_denoise_multi.py -s m -m m -mod model_over -rel y && python3 trans_train_denoise_multi.py -s x -m e -mod model_over -rel y && python3 trans_train_denoise_multi.py -s x -m m -mod model_over -rel y
```
To perform transfer learning with CPC-Feat. Ext and all possible combinations:
```
python3 trans_train_cpc_multi.py -s e -m e -mod model_cpc -rel n && python3 trans_train_cpc_multi.py -s e -m m -mod model_cpc -rel n && python3 trans_train_cpc_multi.py -s m -m e -mod model_cpc -rel n && python3 trans_train_cpc_multi.py -s m -m m -mod model_cpc -rel n && python3 trans_train_cpc_multi.py -s x -m e -mod model_cpc -rel n && python3 trans_train_cpc_multi.py -s x -m m -mod model_cpc -rel n
```
To perform transfer learning with CPC-Full and all possible combinations:
```
python3 trans_train_cpc_multi.py -s e -m e -mod model_cpc -rel y && python3 trans_train_cpc_multi.py -s e -m m -mod model_cpc -rel y && python3 trans_train_cpc_multi.py -s m -m e -mod model_cpc -rel y && python3 trans_train_cpc_multi.py -s m -m m -mod model_cpc -rel y && python3 trans_train_cpc_multi.py -s x -m e -mod model_cpc -rel y && python3 trans_train_cpc_multi.py -s x -m m -mod model_cpc -rel y
```

### Fully trained models

In order to simply use the fully trained models, you can find them in the `models_trans` directory.

## Publication
[Link to paper](https://github.com/juansgomez87/lang-sens-mer/tree/master/ICASSP2021_JSGC.pdf)

For information on our previous work regarding transfer learning from speech to music, please refer to [this repository](https://github.com/juansgomez87/quad-pred).

```
@InProceedings{GomezCanon2021icassp,
    author = {G{\'o}mez-Ca{\~n}{\'o}n, Juan Sebasti{\'a}n  and
              Cano, Estefan{\'i}a and 
              Pandrea, Ana Gabriela and 
              Herrera, Perfecto and 
              G{\'o}mez, Emilia},
    title = {{Language-sensitive Music Emotion Recognition models: are we really there yet?}},
    year = {2021},
    booktitle = {Proceedings of the 46th IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    location = {Toronto, Canada},
    pages = {},
}
```

## Notes
- Our CPC implementation is based on work by [David Tellez](https://github.com/davidtellez/contrastive-predictive-coding). Kudos to him!
- If you want to use the processed datasets (mel-spectrograms + annotations), please contact us: juansebastian.gomez[at]upf.edu

