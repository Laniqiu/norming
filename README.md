# Collecting and Predicting Neurocognitive Norms for Mandarin Chinese

Source code for [Collecting and Predicting Neurocognitive Norms for Mandarin Chinese]

The regression experiments in the paper have been done with traditional static embeddings. You can find them from https://fasttext.cc/docs/en/pretrained-vectors.html, or seek other resources. 


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
