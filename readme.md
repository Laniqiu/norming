# Collecting and Predicting Neurocognitive Norms for Mandarin Chinese

Source code for [Collecting and Predicting Neurocognitive Norms for Mandarin Chinese].

* Note that the regression experiments in the paper have been done with traditional static embeddings. You can find them from https://fasttext.cc/docs/en/pretrained-vectors.html. Alternatively, you can also seek other resources or obtain vectors from language models if needed.
Put the embeddings under the folder: ```data/embs```.


##  Experiment Replication

### Running from Command Prompt

#### Install libraries, modules and packages:

```shell
pip install -r binder/requirements.txt
```

#### Run the scripts:

```shell
python binder/run.py
```

#### Find the evaluation results under the folder ```data/out```:

overall.txt: word and feature spearman correlation
domain.txt: feature correlation by domain
pos.txt: word correlation by part-of-speech


