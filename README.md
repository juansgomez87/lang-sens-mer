# lang-sens-mer

Our previous research showed promising results when transferring features learned from speech to train emotion recognition models for music. 
In this context, we implemented a denoising autoencoder as a pretraining approach to extract features from speech in two languages (English and Mandarin). 
From that, we performed transfer and multi-task learning to predict classes from the arousal-valence space of music emotion. 
We tested and analyzed intra-linguistic and cross-linguistic settings, depending on the language of speech and lyrics of the music. 
This paper presents additional investigation on our approach, which reveals that: (1) performing pretraining with speech in a mixture of languages yields similar results than for specific languages - the pretraining phase appears not to exploit particular language features, (2) the music in Mandarin dataset consistently results in poor classification performance - we found low agreement in annotations, and (3) novel methodologies for representation learning (Contrastive Predictive Coding) may exploit features from both languages (i.e., pretraining on a mixture of languages) and improve classification of music emotions in both languages.
From this study we conclude that more research is still needed to understand what is actually being transferred in these type of contexts. 


## Publication
```
@InProceedings{GomezCanon2021ICASSP,
    author = {Juan Sebasti{\'a}n G{\'o}mez-Ca{\~n}{\'o}n and Estefan{\'i}a Cano and Perfecto Herrera and Emilia G{\'o}mez},
    title = {Language-sensitive Music Emotion Recognition models: are we really there yet?},
    year = {2021},
}
```



