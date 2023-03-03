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


def get_each_emb(model, tokenizer, sen, tgt, strategy="hstack", device="cpu"):
    """
    extract word vec from sen vec
    @param model:
    @param sen:
    @param tgt:
    @param tokenizer:
    @return:
    """

    input_ids = torch.tensor(tokenizer.encode(sen)).to(device)  # .unsqueeze(0)
    tgt_ids = torch.tensor(tokenizer.encode(tgt))[1:-1].to(device)
    # 需要找到target word在tokens中的索引
    ix = (input_ids == tgt_ids[0]).nonzero(as_tuple=True)[0]
    tlen = tgt_ids.size()[0]
    lt = ix[0]
    for ii in ix:
        if (input_ids[ii: ii + tlen] == tgt_ids).all():
            lt = ii
            break
    outputs = model(input_ids.unsqueeze(0))[0].squeeze(axis=0)
    vec = outputs[lt: lt + tlen, :].flatten()

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