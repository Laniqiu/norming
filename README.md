# Collecting and Predicting Neurocognitive Norms for Mandarin Chinese

Source code for [Collecting and Predicting Neurocognitive Norms for Mandarin Chinese]. We would appreciate it if you cite our paper as following:

Le Qiu, Yu-Yin Hsu, Emmanuele Chersoni (2023). Collecting and Predicting Neurocognitive Norms for Mandarin Chinese. In: Proceedings of IWCS.


The regression experiments in the paper have been done with traditional static embeddings. You can find FastText vectors from https://fasttext.cc/docs/en/pretrained-vectors.html, PPMI and Word2Vec ones from https://github.com/Embedding/Chinese-Word-Vectors, or you can seek other resources. 


## Experiment Replication
### Requirement

```shell
pip install -r binder/requirements.txt
```

### Running from Command Prompt
* Note: Please put the embeddings under the folder ```data/embs```. 

```shell
python binder/run.py
```

### Output
Please refer to ```data/out``` folder for evaluation results.

#### overall.txt
Word and feature spearman correlation 
#### domain.txt 
Feature correlation by domain 
#### pos.txt
Word correlation by part-of-speech
