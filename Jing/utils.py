# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 11/1/2023 7:16 pm

"""

import torch
from transformers import BertTokenizer, BertModel
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity as cos
from collections import OrderedDict
import re
import json


from common.setup import adr, logging


# def main(data_path, model_path, sheet=1, model_name="bert-base-chinese"):
def main(fin, model_dir, out_dir, sheet=1, dim=768):
    """

    @param fin:
    @param sheet:
    @return:
    """
    model, tokenizer = load_models(model_dir)

    raw_data = pd.read_excel(fin, sheet_name=sheet)
    targets, senses, sentences = raw_data.targets, raw_data.senses, raw_data.sentences
    assert senses.any()  # senses不可以存在nan值
    assert targets.shape == senses.shape == sentences.shape

    targets = targets.fillna(method="pad").tolist()
    sentences = sentences.fillna("")
    # data = OrderedDict().fromkeys(targets, [])
    PAT = re.compile(r"\d{1,2}\s*\.\s*")
    for _, tgt in enumerate(targets):
        sents = [s for s in PAT.split(sentences[_].replace(" ", "")) if tgt in s]
        out_dir.joinpath(tgt).mkdir(parents=True, exist_ok=True)
        idx = senses[_].split(".")[0]
        fout = out_dir.joinpath(tgt, f"{idx}")

        if not sents:
            logging.warning("\nNo sentences found:\n\ttarget: {}\n\tsense:{}\n".format(tgt, senses[_]))
            continue
        vec_list = [process_each_sent(model, sen, tgt, tokenizer) for sen in sents]
        sense_vec = torch.vstack(vec_list).mean(axis=0).unsqueeze(0)
        torch.save(sense_vec, fout)





def process_each_sent(model, sen, tgt, tokenizer):
    """
    extract target word vector from each sen vector
    @param model:
    @param sen:
    @param tgt:
    @param tokenizer:
    @return:
    """
    input_ids = torch.tensor(tokenizer.encode(sen))  # .unsqueeze(0)
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    logging.info("tokenized:\n\t{}".format(tokens))
    # 需要找到target word在tokens中的索引
    lt = tokens.index(tgt[0])
    for i in range(tokens.count(tgt[0])):
        # rt = lt + len(tgt)
        mm = "".join(tokens[lt: lt + len(tgt)])
        if mm == tgt:
            logging.info("target word: {}\nmatched word:{}".format(tgt, mm))
            break
        lt = tokens[lt + 1:].index(tgt[0])
    outputs = model(input_ids.unsqueeze(0))[0].squeeze(axis=0)
    vec = outputs[lt:lt + len(tgt), :]
    return vec


def load_models(model_dir, version="bert-base-chinese"):
    """
    load tokenizer and model
    @param model_name:
    @param tokenizer_name:
    @return:
    """
    logging.info("loading models from {}, version:{}".format(model_dir, version))

    tokenizer = BertTokenizer.from_pretrained(model_dir.joinpath(version))  # 加载base模型的对应的切词器
    model = BertModel.from_pretrained(model_dir.joinpath(version))
    return model, tokenizer


def load_data(fin, fout, sheet=1):

    raw_data = pd.read_excel(fin, sheet_name=sheet)

    # 构造成 target: (sense1, sents1), ...
    targets, senses, sentences = raw_data.targets, raw_data.senses, raw_data.sentences
    assert senses.any()  # 不存在nan值
    assert targets.shape == senses.shape == sentences.shape

    targets = targets.fillna(method="pad")
    sentences = sentences.fillna("")
    data = OrderedDict().fromkeys(targets, [])
    PAT = re.compile(r"\d{1,2}\s*\.\s*")
    for _, tgt in enumerate(targets):
        sense = senses[_]
        sents = [s for s in PAT.split(sentences[_].replace(" ", "")) if s]

        if not sents:
            logging.warning("\nNo sentences found:\n\ttarget: {}\n\tsense:{}\n".format(tgt, sense))

        data[tgt].append((sense, sents))
    with open(fout, "w") as fw:
        json.dump(data, fw, ensure_ascii=False)
    logging.info("data dumped at {}".format(fout))

    return data


def sim_report(m1, m2):
    return cos(m1, m2)



if __name__ == "__main__":
    _root = Path(adr)
    tmp_dir = Path("/content/tmp")

    fin = _root.joinpath("dough/evaluation-target-candidates.xlsx")
    out_dir = tmp_dir.joinpath("vectors")


    # data = load_data(fin, fout)
    # model, tokenizer = load_models(tmp_dir)
    main(fin, tmp_dir, out_dir)
