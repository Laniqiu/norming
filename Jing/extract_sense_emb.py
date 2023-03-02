# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 11/1/2023 7:16 pm

"""
import pandas as pd
import re

import sys
sys.path.insert(0, "/home/qiule/pies/")


from common.setup import logging, adr, tmp_dir
from utils import load_models, collect_embs, save_vecs


def pilot(fin, out_dir, model_path, sheet=1):
    """
    对target word的每一个sense取平均word emb
    @param fin:
    @param out_dir:
    @param model_path:
    @param sheet:
    @param temp_dir:
    @return:
    """
    logging.info("loading models from {}".format(model_path))
    model, tokenizer = load_models(model_path)

    logging.info("loading data from {}".format(fin))
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
        fname = senses[_].split(".")[0]  # 当前意义名称
        fout = temp_dir.joinpath(tgt, f"{fname}")
        temp_dir.joinpath(tgt).mkdir(parents=True, exist_ok=True)
        sents_vec = collect_embs(tgt, sents, model, tokenizer)
        sense_vec = sents_vec.mean(axis=0).unsqueeze(0)
        save_vecs(sense_vec, fout)
        logging.info(f"temporarily saving at {fout}")


def collect(files, model_path, temp_dir):
    """
    @param files: Path object, .xlsx files to be processed
    @param model_path: Path object
    @return:
    """
    logging.info("loading models from {}".format(model_path.name))
    model, tokenizer = load_models(str(model_path))

    if not temp_dir.exists():
        temp_dir.mkdir(exist_ok=True)

    for f in files:
        this_data = pd.read_excel(f, sheet_name=None)
        for tgt, df in this_data.items():
            tgt = tgt.strip()
            for i in range(df.shape[0]):
                this_sents = [s.strip().replace(" ", "") for s in df.iloc[i, 1:].dropna() if tgt in s]
                # 对例句处理
                if not this_sents:
                    logging.warning("\n No examples found for {}".format(tgt))
                    continue
                fout = temp_dir.joinpath(f"{tgt}_{i+1}")
                sents_vec = collect_embs(tgt, this_sents, model, tokenizer)
                sense_vec = sents_vec.mean(axis=0).unsqueeze(0)
                save_vecs(sense_vec, fout)
                logging.info(f"sense embedding temporarily saving at {fout}")


if __name__ == "__main__":
    model_version = "chinese-roberta-wwm-ext"
    model_path = tmp_dir.joinpath(model_version)

    temp_dir = tmp_dir.joinpath("svecs")  # 临时数据保存路径

    # 这是之前pilot study抽取sense的代码
    # pilot(adr.joinpath("Jing/evaluation-target-candidates.xlsx"),  # 输入数据
    #      adr.joinpath("Jing"),   # 输出最终保存路径
    #      model_path,  # 模型路径
    #      )

    # final project抽取sense
    _dir = adr.joinpath("Jing")
    collect(_dir.joinpath("checking-targets").glob("*.xlsx"),
            adr.joinpath("lfs/chinese_roberta_wwm_ext_pytorch"),  # lm模型路径
            _dir.joinpath("sens_embs")  # 单义词sense emb保存目录
            )



