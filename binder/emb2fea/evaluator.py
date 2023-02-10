# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 20/1/2023 7:48 pm
1. spearmanr score by word & by feature
2. output top & bottom words in terms of MAE/ MSE
3. spearmanr cor between MAE (or MSE) and freq
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from pathlib import Path

from common.setup import *
from common.io_utils import general_reader, general_writer


def spr_words_feas(Y_test, Y_pred):
    """
    spearman cof across words & features
    @param Y_test: array (n_samples, n_features)
    @param Y_pred: array (n_samples, n_features)
    @return:
    """
    sp_w, sp_f = [], []
    if Y_test.shape != Y_pred.shape:
        assert False, "The size of the prediction array Y and of the test array Y are different."

    wn, fn = Y_test.shape
    for i in range(fn):
        var = spearmanr(Y_test[:, i], Y_pred[:, i])[0]
        sp_f.append(var)

    for i in range(wn):
        var = spearmanr(Y_test[i], Y_pred[i])[0]
        sp_w.append(var)
    return np.array(sp_f, dtype=float), np.array(sp_w, dtype=float)


def vis(arr, words, num):
    ss = np.argsort(arr)
    tt = zip(ss[:num], ss[-num:][::-1])
    out = []
    for i1, i2 in tt:
        tw, bw = words[i1], words[i2]  # list类型
        tp, bt = arr[i1], arr[i2]
        ll = map(str, [tw[0], tw[1], tp, bw[0], bw[1], bt])
        line = "\t".join(ll)
        out.append(line)
    return out


def main(fpth, in_dir, out_dir, gpat="gold", ppat="predict", num=10):
    """

    @param fpth: path to the ratings
    @param in_dir:
    @param out_dir: output dir
    @param gpat:
    @param ppat:
    @param num: visualize the top(bottom) num
    @return:
    """

    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    pths = in_dir.glob("*{}.npy".format(gpat))

    df = pd.read_excel(fpth)
    out1 = ["Model\tRegressor\tWord Correlation\tFeature Correlation\tMAE-Freq\tMSE-Freq\n"]
    out2 = ["Model\tRegressor\tTop(MAE)\t\t\tBottom(MAE)\t\t\tTop(MSE)\t\t\tBottom(MSE)\t\t\n"]

    for gpth in sorted(pths):
        logging.info("Processing {}".format(gpth))
        # load saved output & gt
        model, reg, _ = gpth.name.split("_")  # language model, regressor
        ppth = gpth.parent.joinpath(gpth.name.replace(gpat, ppat))
        gt = np.load(gpth).squeeze()
        pred = np.load(ppth).squeeze()
        sp_f, sp_w = spr_words_feas(gt, pred)  # spearmanr by word & by fea

        # load freq info
        wpth = in_dir.joinpath("{}_words.txt".format(model))
        words = [e.strip().split("\t") for e in general_reader(wpth)]
        fs = [f for _, f in enumerate(df["BCC(log10)"]) if [df["EngWords"][_], df["words"][_]] in words]  # freq
        fs = np.array(fs, dtype=float)

        mae = mean_absolute_error(gt.T, pred.T, multioutput="raw_values")
        mse = mean_squared_error(gt.T, pred.T, multioutput="raw_values")
        spr_a = spearmanr(mae, fs)[0]
        spr_s = spearmanr(mse, fs)[0]

        ll = map(str, [model, reg, sp_w.mean(), sp_f.mean(), spr_a, spr_s])
        line = "\t".join(ll) + "\n"
        out1.append(line)

        va = vis(mae, words, num)
        vs = vis(mse, words, num)
        for idx, ach in enumerate(va):
            sch = vs[idx]
            out2.append("{}\t{}\t{}\t{}\n".format(model, reg, ach, sch))
    fout1 = out_dir.joinpath("cor_fea.txt")
    fout2 = out_dir.joinpath("cor_freq.txt")
    general_writer(out1, fout1)
    general_writer(out2, fout2)



# if __name__ == '__main__':
#     _ddir = adr.joinpath("binder")
#
#     fpth = _ddir.joinpath("Copy of meanRating_July1.xlsx")
#     wpth = _ddir.joinpath("out4/cc.zh.300_words.txt")
#     gpat, ppat = "gold", "predict"
#     pths = _ddir.joinpath("out4").glob("*{}.npy".format(gpat))
#     out_dir = _ddir.joinpath("out4")
#
#     num = 20
#
#     main()

















