# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 7/2/2023 10:36 am

"""
import sys
sys.path.insert(0, '/Users/laniqiu/Drive/pies')
sys.path.insert(0, "/home/qiule/drive/pies")

import pandas as pd

from common.setup import adr

_path = adr.joinpath("binder/repeated_words")
res = {}
# 統計均分
for f in sorted(_path.glob("*.xlsx")):
    if "cal" in f.name:
        continue
    wi1, wi2, _, w1, w2 = f.stem.replace(" ", "-").replace("，", "-").split("-")
    each = pd.read_excel(f)
    each.replace("无关", 0, inplace=True)
    scr1, scr2 = each.iloc[:, 1:69].values.astype(dtype=int).T, \
        each.iloc[:, 69:-1].values.astype(dtype=int).T

    res[(wi1, w1)] = scr1.mean(axis=1)
    res[(wi2, w2)] = scr2.mean(axis=1)

# 替換原表格中的值
ori_file = adr.joinpath("binder/meanRating_July1.xlsx")
data = pd.read_excel(ori_file)

wids, ws = zip(*res.keys())

for key, value in res.items():
    i, w = key
    if w == "砂紙":  # todo 暫時不管砂紙
        continue
    col_idx = int(i) + 4
    data.words[col_idx] = w
    data.length[col_idx] = len(w)
    data.iloc[col_idx, 11: 11 + 68 + 1] = value

fout = adr.joinpath("binder/new_ratings.xlsx")
data.to_excel(fout, index_label=False, index=False)
if __name__ == "__main__":
    pass