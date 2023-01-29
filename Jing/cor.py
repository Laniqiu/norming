# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 24/1/2023 12:21 pm
计算correlation
"""
import pandas as pd
import numpy as np
from scipy.stats.mstats import spearmanr
from sklearn.metrics import cohen_kappa_score, precision_score
from collections import OrderedDict

from common.setup import *
from common.io_utils import general_reader


def merge_annotations(an1, an2, preds, fout):
    """
    @param an1: 人工标注分数文件
    @param an2:
    @param preds: 模型预测的分数文件
    @param fout:
    @return:
    """
    # merge data from 2 annotators
    merged = {}
    for f1, f2, ppth in zip(sorted(an1), sorted(an2), sorted(preds)):
        assert f1.stem == f2.stem == ppth.stem.split("_")[0]

        # ppth = bdir.joinpath(f1.stem)
        d1 = pd.read_excel(f1)
        d2 = pd.read_excel(f2)

        pt = get_preds(ppth, d1.iloc[:, 0], d1.iloc[:, 1])

        # 处理异常值（NA, P)， NA: 无法判断; P: 歧义"""
        score1 = d1.iloc[:, 3].replace("P", np.nan).replace("NA", np.nan)\
            .replace(-1, np.nan).replace(0, np.nan)
        score2 = d2.iloc[:, 4].replace("P", np.nan).replace("NA", np.nan)\
            .replace(-1, np.nan).replace(0, np.nan)

        # vis()

        d1.iloc[:, 3] = score1
        d1.iloc[:, 4] = score2
        d1["Bert"] = pt

        merged[f1.stem] = d1
    with pd.ExcelWriter(fout) as writer:
        for k, v in merged.items():
            v.to_excel(writer, sheet_name=str(k), index=False)


def vis(score1, score2):
    print("a1, nan值：", np.where(score1.values == -1)[0].shape[0], end=", ")
    print("a2, nan值：", np.where(score2.values == -1)[0].shape[0], end=", ")

    print("a1, P值：", np.where(score1.values == 0)[0].shape[0], end=", ")
    print("a2, P值：", np.where(score2.values == 0)[0].shape[0], end=", ")

    print("a1不等于a2：", np.where(score1.values != score2.values)[0].shape[0])


def get_preds(ppth, years, idx):
    pdata = {}
    with open(ppth, "r") as fr:
        for i, line in enumerate(fr):
            if i == 0:
                continue
            ss, yy = line.strip().split("\t")
            if yy not in pdata:
                pdata[yy] = []
            pdata[yy].append(ss)

    pt = []
    for _, yr in enumerate(years):
        key = str(yr)
        if key not in pdata:
            continue
        ix = idx[_]
        if np.isnan(ix):
            val = np.nan
        else:

            val = int(pdata[key][int(ix)])

        pt.append(val)

    return pt


def compute_cor(fin, fout1, fout2):
    """
    计算人工评分与模型评分的相关性，16个target words的相关性以及总体相关性
    @return:
    """
    res = OrderedDict()
    rater = OrderedDict()
    df = pd.read_excel(fin, sheet_name=None)
    ann, pred = [], []
    r1, r2 = [], []

    for tgt, data in df.items():

        scores = data.iloc[:, 3: 6].dropna()
        scores.reset_index(drop=True, inplace=True)
        a1, a2, pt = scores.iloc[:, 0], scores.iloc[:, 1], scores.iloc[:, 2]

        rater[tgt] = np.round(cohen_kappa_score(a1, a2), 4)
        r1 += a1.tolist()
        r2 += a2.tolist()

        ix = np.where(a1 == a2)[0]  # 人工评分一致的
        res[tgt] = np.round(cohen_kappa_score(a1[ix], pt[ix]), 4)

        ann += a1[ix].tolist()
        pred += pt[ix].tolist()

    ann, pred = np.array(ann), np.array(pred)
    res["All"] = np.round(cohen_kappa_score(ann, pred), 4)
    rater["ALL"] = np.round(cohen_kappa_score(r1, r2), 4)
    dr = pd.DataFrame.from_dict(rater, orient="index")
    de = pd.DataFrame.from_dict(res, orient="index")

    dr.to_csv(fout1, sep="\t")
    de.to_csv(fout2, sep="\t")


def compute_pre(fin, fout):
    res = OrderedDict()
    df = pd.read_excel(fin, sheet_name=None)
    ann, pred = [], []
    for tgt, data in df.items():

        scores = data.iloc[:, 3: 6].dropna()
        scores.reset_index(drop=True, inplace=True)
        a1, a2, pt = scores.iloc[:, 0], scores.iloc[:, 1], scores.iloc[:, 2]

        ix = np.where(a1 == a2)[0]  # 人工评分一致的
        res[tgt] = np.round(precision_score(a1[ix], pt[ix], average="micro"), 4)

        ann += a1[ix].tolist()
        pred += pt[ix].tolist()

    ann, pred = np.array(ann), np.array(pred)
    res["All"] = np.round(precision_score(ann, pred, average="micro"), 4)

    de = pd.DataFrame.from_dict(res, orient="index")
    de.to_csv(fout, sep="\t")


def check_sea():
    fpth = adr.joinpath("Jing/annotator1/下海.xlsx")
    fdir = adr.joinpath("Jing/data_filtered/下海")
    df = pd.read_excel(fpth)
    poss = []
    for _, yr in enumerate(df.iloc[:, 0]):
        ppth = fdir.joinpath("{}下海.txt".format(yr))
        sent = df.iloc[:, 2][_]
        sents = [s.strip() for s in general_reader(ppth)]
        if sent not in sents:
            print(df.iloc[:, 1][_])
            print(sent)
            val = np.nan
        else:
            val = sents.index(sent)
        poss.append(val)

    df.iloc[:, 1] = poss
    df.to_excel(fpth, index=False)


if __name__ == '__main__':
    an1 = adr.joinpath("Jing/annotator1").glob("*.xlsx")
    an2 = adr.joinpath("Jing/annotator2").glob("*.xlsx")
    bdir = adr.joinpath("Jing/bert_prediction").glob("*.txt")

    fout = adr.joinpath("Jing/merged_anns.xlsx")

    # merge_annotations(an1, an2, bdir, fout)
    # check_sea()  #  下海的数据对不上，校正

    # compute_cor(adr.joinpath("Jing/merged_anns.xlsx"),
    #             adr.joinpath("Jing/cor_rater.txt"),
    #             adr.joinpath("Jing/cor_res.txt")
    #             )

    compute_pre(adr.joinpath("Jing/merged_anns.xlsx"),
                adr.joinpath("Jing/precision.txt")
                )



















