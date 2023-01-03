# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 1/1/2023 9:54 pm
检查spearman
"""
import numpy as np
import pandas as pd

from common import logging


rdict = {"0": "Linear", "1": "Lasso",
         "2": "Ridge", "3": "RandomForest", "4": "MLP"}

columns = ["Model", "Regressor", "Word Correlation", "Feature Correlation"]
# indexes = ["cc.zh.300.vec", "sgns.wiki.word", "wiki.zh.aligh.vec", "wiki.zh.vec"] * len(rdict)


def main():
    outt = []
    for this_file in files:

        logging.info("loading from {}".format(this_file.name))
        vec, reg, _, st = this_file.stem.split("_")
        this_data = np.load(this_file)

        that_file = this_file.parent.joinpath(this_file.name.replace("_fea.npy", "_word.npy"))
        logging.info("loading from {}".format(that_file.name))
        that_data = np.load(that_file)

        row = {"Model": this_file.name.split("_", 1)[0],
               "Regressor": rdict[reg],
               "Word Correlation": np.round(that_data.mean(), decimals=4),
               "Feature Correlation": np.round(this_data.mean(), decimals=4)}

        outt.append(row)
    df = pd.DataFrame(outt)
    df.to_csv(fout, sep="\t")


if __name__ == "__main__":
    from pathlib import Path
    fpth = "/Users/laniqiu/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/assignments/binder/spearman"
    fout = "/Users/laniqiu/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/assignments/binder/" \
           "spearman.txt"
    files = sorted(Path(fpth).glob("*_fea.npy"))
    main()





