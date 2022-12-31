"""
calculate mi (replication on binder data?)
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, mutual_info_score, adjusted_mutual_info_score
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from pandas import Series, DataFrame
from copy import deepcopy
from itertools import combinations
from collections import OrderedDict
import logging

try:
    from google.colab import drive
    _root = "/content/drive/MyDrive/"
except:
    _root = "/Users/laniqiu/My Drive/"

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def excel_to_df(fpth, tgt_st):
    df = pd.read_excel(fpth, sheet_name=tgt_st)
    return df


def all_about_mi(df, handle_nans='pad', nan='na', spth="save.png"):

    # load indices
    attr_names = df[1]['Attribute'].values.tolist()
    sorted_attrs = df[1]['Sorted'].values.tolist()

    # load mean ratings according to the headline
    ratings = df[0][attr_names].copy()
    # repalce missing values to np.nan
    ratings.iloc[:, :].replace(nan, np.nan, inplace=True)

    n = ratings.shape[1]

    logging.info("compute pairwise mi")
    # compute pairwise mi (filling or deleting nan values)
    mis = cal_bt_attrs(n, ratings, method=handle_nans)

    # in-group pairwise mi
    # components = df[1]['Component'].values.tolist()
    # comp_attr = df[1]['Component-Attr'].values.tolist()
    groups = df[1]['Group'].dropna().values.tolist()
    grp_attr = df[1]["Group-Attr"].dropna().tolist()
    logging.info("compute in-group pairwise mi")
    gr_scores = cal_pairwise_mi(groups, grp_attr, attr_names, mis)

    logging.info("draw heatmap")
    generate_heatmap(mis.copy(), n, attr_names, sorted_attrs, spth)

    visualize_for_me(mis.copy(), attr_names, gr_scores, top=10)


def cal_pairwise_mi(groups, grp_attr, attr_names, mis):
    """

    """
    ga_scores = OrderedDict()
    for gr in list(set(groups)):
        start = groups.index(gr)
        end = start + groups.count(gr)
        comb = combinations(grp_attr[start:end], 2)
        value = []
        for each in list(comb):
            ix1, ix2 = attr_names.index(each[0]), attr_names.index(each[1])
            idx = (ix1, ix2) if ix1 < ix2 else (ix2, ix1)
            _mi = mis[idx]
            value.append(_mi)
        ga_scores[gr] = np.array(value).mean()
    return ga_scores


def visualize_for_me(mis, attr_names, ga_scores, top=10):
    """
    personal visualization
    """
    print("mean: ", mis.mean())
    print("max: ", mis.max())
    print("pairwise mi reported in Binder et al's ")
    print(np.where(mis == mis.max()))
    print(mis[6, 18])
    print(mis[44, 45])
    print(mis[32, 36])
    print(mis[27, 28])
    print(mis[46, 48])
    print("the top:")
    ft = mis.flatten()
    aa = ft[np.argsort(-ft)][:top]
    for i in range(top):
        idx = np.where(mis==aa[i])
        print('{}-{}\t{}'.format(attr_names[int(idx[0])], attr_names[int(idx[1])], aa[i]))
    print("in-group pairwise mi")
    for k, v in ga_scores.items():
        print('{}\t{}'.format(k, v))


def cal_bt_attrs(n, ratings, method):
    """
    calculate mi score between each attribute pair
    :param n:
    :param ratings:
    :return:
    """
    mis = np.zeros((n, n))
    # discretizer
    est = KBinsDiscretizer(n_bins=6, encode='ordinal', strategy='kmeans')
    for i in np.arange(n):
        for j in np.arange(i + 1, n):
            xy = ratings.iloc[:, [i, j]].copy()
            if method:
                xy.fillna(method=method, inplace=True)
            else:
                xy.dropna(axis=0, how='any', inplace=True)
            dxy = est.fit_transform(xy)
            mis[i, j] = normalized_mutual_info_score(dxy[:, 0], dxy[:, 1])
            # mis[i, j] = mutual_info_score(dxy[:, 0], dxy[:, 1])
    return mis


def generate_heatmap(mis, n, headline, sort_head, spth):
    """
    mis: matrix of mi scores
    headline: head titile of mi scores
    sort_head?
    spth: save path of heatmap
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.figure(dpi=200)
    sns.set(font_scale=1)

    """convert mis, a triangle matrix, to a regular one for visualization """
    fmis = mis + mis.T - np.diag(mis.diagonal())
    for i in np.arange(n):
        fmis[i, i] = 1
    head_idx = [sort_head.index(e) for e in headline]
    df = pd.DataFrame(fmis, index=head_idx, columns=head_idx)
    # re-order of the data to generate a heatmap
    df.sort_index(axis=0, inplace=True)
    df.sort_index(axis=1, inplace=True)
    df.index = Series(sort_head)
    df.columns = Series(sort_head)
    # draw a heatmap
    ax = sns.heatmap(df, cmap="YlGnBu", xticklabels=True, yticklabels=True)
    plt.tick_params(axis="both", labelsize=5)
    plt.savefig(spth)
    logging.info("save pic at %s" % spth)
    plt.show()



if __name__ == '__main__':
    fpth = os.path.join(_root, "dough/WordSet1_Ratings.xlsx")
    df = excel_to_df(fpth, [0, 1])
    all_about_mi(df, spth=os.path.join(_root, "dough/save.png"))
