"""
"""
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
from collections import OrderedDict

from common.setup import *


def main(fpth, fout, minr=20, maxr=31, chinese=True):
    cwords, ewords, ratings = load_data(fpth, chinese=chinese)

    outt = {}
    for n in range(minr, maxr):
        logging.info("n_cluster={}".format(n))
        kmeans = KMeans(n_clusters=n).fit(ratings)  # default..
        each = OrderedDict()
        llen = 0
        for i in np.arange(n):
            idx = np.where(kmeans.labels_ == i)
            clr = cwords.iloc[idx].values
            elr = ewords.iloc[idx].values
            each[i] = clr
            if chinese:  # 是否加上英语单词，对于中文适用
                each[str(i)] = elr
            llen = max(clr.shape[0], llen)

        for k, v in each.items():
            each[k] = np.pad(v, (0, llen - v.shape[0]),
                             mode="constant", constant_values=np.nan)
        outt[n] = pd.DataFrame.from_dict(each)
    with pd.ExcelWriter(fout) as writer:
        for k, v in outt.items():
            v.to_excel(writer, sheet_name=str(k), index=False)


def load_data(fpth, chinese=True):
    """

    @param fpth:
    @param chinese: if the dataset is in Ch (from Dr. Hsu), then add eng words too4
    @return:
    """
    logging.info("load data ...")
    df = pd.read_excel(fpth)
    # 加载评分数据
    ratings = df.iloc[5:, 11:].values.astype(float)
    if chinese:
        ewords, cwords = df.iloc[5:, 2], df.iloc[5:, 3]
    else:
        ewords, cwords = None, None  # todo change if it is eng data
    if True in np.isnan(ratings):  # 检查有无nan值
        logging.warning("Nan detected, exit ...")
        exit()
    return cwords, ewords, ratings


if __name__ == '__main__':

    fpth = Path("ratings_update_Feb24.xlsx")
    fout = Path("kmeans.xlsx")

    main(fpth, fout, chinese=True)
