# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 12/1/2023 8:57 pm

"""
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity as cos
import numpy as np

def load_models(model_path):
    """
    load tokenizer and model
    @param model_name:
    @param tokenizer_name:
    @return:
    """
    tokenizer = BertTokenizer.from_pretrained(model_path)  # 加载base模型的对应的切词器
    model = BertModel.from_pretrained(model_path)
    return model, tokenizer


def get_each_emb(model, tokenizer, sen, tgt, strategy="hstack"):
    """
    extract word vec from sen vec
    @param model:
    @param sen:
    @param tgt:
    @param tokenizer:
    @return:
    """
    input_ids = torch.tensor(tokenizer.encode(sen))  # .unsqueeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    # 需要找到target word在tokens中的索引
    lt = tokens.index(tgt[0])
    for i in range(tokens.count(tgt[0])):
        mm = "".join(tokens[lt: lt + len(tgt)])
        if mm == tgt:
            break
        lt = tokens[lt + 1:].index(tgt[0])
    outputs = model(input_ids.unsqueeze(0))[0].squeeze(axis=0)
    # if strategy == "hstack":
    vec = outputs[lt:lt + len(tgt), :].flatten()

    return vec

def collect_embs(tgt, sents, model, tokenizer, strategy="hstack"):
    """
    给定一组句子，获得target word的所有emb
    @param tgt:
    @param sents:
    @param model:
    @param tokenizer:
    @return:
    """
    vec_list = [get_each_emb(model, tokenizer, sen, tgt, strategy) for sen in sents]
    return torch.vstack(vec_list)

def save_vecs(vec, fout):
    torch.save(vec, fout)

def load_vecs(fin):
    return torch.load(fin)


def sim_report(m1, m2, decimals=6):
    score = cos(np.asarray(m1), np.asarray(m2))
    return np.round(score, decimals=decimals)