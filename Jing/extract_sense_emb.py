# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 11/1/2023 7:16 pm

"""
import pandas as pd
import re

from common.setup import logging, adr, tmp_dir
from utils import load_models, collect_embs, save_vecs


def main(fin, out_dir, model, tokenizer, sheet=1):

    """
    对target word的每一个sense取平均word emb
    @param fin:
    @param sheet:
    @return:
    """

    raw_data = pd.read_excel(fin, sheet_name=sheet)
    targets, senses, sentences = raw_data.targets, raw_data.senses, raw_data.sentences
    assert senses.any()  # senses不可以存在nan值
    assert targets.shape == senses.shape == sentences.shape

    targets = targets.fillna(method="pad").tolist()
    sentences = sentences.fillna("")
    PAT = re.compile(r"\d{1,2}\s*\.\s*")
    for _, tgt in enumerate(targets):
        sents = [s for s in PAT.split(sentences[_].replace(" ", "")) if tgt in s]
        if not sents:
            logging.warning("\nNo sentences found:\n\ttarget: {}\n\tsense:{}\n".format(tgt, senses[_]))
            continue

        fname = senses[_].split(".")[0]
        fout = out_dir.joinpath(tgt, f"{fname}")
        out_dir.joinpath(tgt).mkdir(parents=True, exist_ok=True)
        sense_vec = collect_embs(tgt, sents, model, tokenizer).mean(axis=0).unsqueeze(0)
        save_vecs(sense_vec, fout)
        logging.info(f"average word embedding save at {fout}")


if __name__ == "__main__":
    fin = adr.joinpath("Jing/evaluation-target-candidates.xlsx")  # 输入数据
    out_dir = adr.joinpath("svectors")  # 最终保存路径

    model_version = "chinese-roberta-wwm-ext"
    model_path = tmp_dir.joinpath(model_version)
    model, tokenizer = load_models(model_path)

    main(fin, out_dir, model, tokenizer)


