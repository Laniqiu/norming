# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 27/1/2023 10:13 am

"""
import pandas as pd
import numpy as np
from collections import OrderedDict


def extract_err_sents():
    """
    將錯誤句子抽出來，忽略NA和P以人工評分不一致的
    @return:
    """
    df = pd.read_excel(fpth, sheet_name=None)
    out = OrderedDict()
    for tgt, that_data in df.items():
        ix = np.where(that_data.iloc[:, 3] != that_data.iloc[:, 4])[0]
        data = that_data.drop(index=ix)  # 去除空值
        data.reset_index(drop=True, inplace=True)
        ix = np.where(data.iloc[:, 3] == data.iloc[:, 5])[0]
        data = data.drop(index=ix)  # 去除正確預測句子
        data.reset_index(drop=True, inplace=True)
        data["References"] = that_data.Reference
        data = data.drop(data.columns[[3]], axis=1)
        out[tgt] = data
    with pd.ExcelWriter(fout) as writer:
        for k, v in out.items():
            v.to_excel(writer, sheet_name=k, index=False)


def stat(fin, fout):
    """
    統計錯誤
    @param fin:
    @param fout:
    @return:
    """
    out = []
    df = pd.read_excel(fin, sheet_name=None)
    for tgt, that_data in df.items():
        # 無效
        a1, a2, pt = that_data.iloc[:, 3], that_data.iloc[:, 4], that_data.iloc[:, 5]
        ix = np.where(a1 == a2)[0]
        cc = np.where(a1[ix] == pt[ix])[0].shape[0]
        total = that_data.shape[0]  # total sample
        vd = ix.shape[0]  # valid sample

        ic = vd - cc
        iv = total - vd
        out.append([tgt, cc, ic, vd, iv, total])
    head = ["Word", "Correctly Classified", "Incorrectly Classified", "Valid Samples",
            "Invalid Samples", "Total"]
    out.insert(0, head)

    outt = pd.DataFrame(out)
    final = [["Total"]]
    for i in range(1, 6):
        final.append([outt.iloc[1:, i].values.sum()])

    outt.to_excel(fout, index=False)




if __name__ == '__main__':
    from common.setup import *

    fpth = adr.joinpath("Jing/merged_anns.xlsx")
    fout = adr.joinpath("Jing/error_sents.xlsx")

    # extract_err_sents()

    stat(adr.joinpath("Jing/merged_anns.xlsx"),
         adr.joinpath("Jing/stats.xlsx")
    )


