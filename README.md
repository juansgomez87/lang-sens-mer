# Language-sensitive Music Emotion Recognition models: are we really there yet?

Our previous research showed promising results when transferring features learned from speech to train emotion recognition models for music. 
In this context, we implemented a denoising autoencoder as a pretraining approach to extract features from speech in two languages (English and Mandarin). 
From that, we performed transfer and multi-task learning to predict classes from the arousal-valence space of music emotion. 
We tested and analyzed intra-linguistic and cross-linguistic settings, depending on the language of speech and lyrics of the music. 
This paper presents additional investigation on our approach, which reveals that: (1) performing pretraining with speech in a mixture of languages yields similar results than for specific languages - the pretraining phase appears not to exploit particular language features, (2) the music in Mandarin dataset consistently results in poor classification performance - we found low agreement in annotations, and (3) novel methodologies for representation learning (Contrastive Predictive Coding) may exploit features from both languages (i.e., pretraining on a mixture of languages) and improve classification of music emotions in both languages.
From this study we conclude that more research is still needed to understand what is actually being transferred in these type of contexts. 

## Usage
TODO

## Publication
[Link to paper](https://github.com/juansgomez87/lang-sens-mer/tree/master/ICASSP2021_JSGC.pdf)

```
@InProceedings{GomezCanon2021,
    author = {G{\'o}mez-Ca{\~n}{\'o}n, Juan Sebasti{\'a}n  and
              Cano, Estefan{\'i}a and 
              Pandrea, Ana Gabriela and 
              Herrera, Perfecto and 
              G{\'o}mez, Emilia},
    title = {{Language-sensitive Music Emotion Recognition models: are we really there yet?}},
    year = {2021},
    booktitle = {Proceedings of the International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
    location = {Toronto, Canada},
    pages = {},
}
```

## Notes
Our CPC implementation is based on work by [David Tellez](https://github.com/davidtellez/contrastive-predictive-coding). Kudos to him!
