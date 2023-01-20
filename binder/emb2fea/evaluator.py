# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 7/1/2023 7:38 pm
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from common.io_utils import general_reader, general_writer
from common.setup import *


def show_mse(fdir="binder/sps", out_name="spr_show.txt", pat="gold", ptt="predict", ):
    """
    输出top 10和bottom 10 mse words
    @return:
    """
    _ddir = adr.joinpath(fdir)
    words = general_reader(_ddir.joinpath("cc.zh.300_words.txt"))
    words = [w.strip() for w in words if not w.startswith("_")]  # 过滤_matrix_

    fname = _ddir.parent.joinpath("Copy of meanRating_July1.xlsx")  # 原文件名
    df = pd.read_excel(fname)
    zipped = list(zip(df.EngWords, df.words))
    ws = [(e, c) for e, c in zipped if c in words]
    out = ["model\tregressor\ttop-10\tword\tscore\tbottom-10\tword\tscore\n"]

    model = "cc.300"
    for gold in _ddir.glob("*{}.npy".format(pat)):  # 匹配模式
        print(gold)
        tag, _ = gold.name.split("_")
        pred = _ddir.joinpath(gold.name.replace(pat, ptt))
        ga, pa = np.load(gold), np.load(pred)  # 原来保存的输出
        scores = []
        for i in range(ga.shape[0]):
            mse = mean_squared_error(ga[i], pa[i], multioutput='raw_values')
            scores.append(mse)
        scores = np.array(scores).reshape(-1)
        ss = np.argsort(scores)
        tt = zip(ss[:10], ss[-10:][::-1])
        for t, b in tt:
            line = [model, tag, ws[t][0], ws[t][1], str(scores[t]), ws[b][0], ws[b][1], str(scores[b])]
            out.append("\t".join(line) + "\n")
    fout = _ddir.joinpath(out_name)
    general_writer(out, fout)


def mae_freq():
    """
    计算每个词的mae和freq之间的关系
    @return:
    """
    pass


def check_spr(files, fout, that_name, this_name, rdict):
    outt = []
    for this_file in sorted(files):
        logging.info("loading from {}".format(this_file.name))
        vec, reg, _, st = this_file.stem.split("_")
        this_data = np.load(this_file)  # spearman on features

        that_file = this_file.parent.joinpath(this_file.name.replace(that_name, this_name))
        logging.info("loading from {}".format(that_file.name))
        that_data = np.load(that_file)  # spearman on words

        row = {"Model": this_file.name.split("_", 1)[0],
               "Regressor": rdict[reg],
               "Word Correlation": np.round(that_data.mean(), decimals=4),
               "Feature Correlation": np.round(this_data.mean(), decimals=4)}

        outt.append(row)
    df = pd.DataFrame(outt)
    df.to_csv(fout, sep="\t")

def spr_overall():
    """
    spearmanr correlation by words & by features
    @return:
    """



