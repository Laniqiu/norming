
import pandas as pd
from common.setup import logging

from binder.utils.mi_on_binder import cal_pairwise_mi, cal_bt_attrs, \
    generate_heatmap, visualize_for_me


def main(fpth, spth, handle_nans="pad"):
    """
    @param fpth: path the rating file
    @param spth: path to save the heatmap
    @param handle_nans: how to deal with nan values
    @return:
    """
    df = pd.read_excel(fpth)
    ratings = df.iloc[5:, 11:]
    attr_names = df.iloc[3, 11:].to_list()
    sorted_attrs = attr_names

    n = ratings.shape[1]
    logging.info("compute pairwise mi")
    mis = cal_bt_attrs(n, ratings, method=handle_nans)
    logging.info("compute in-group pairwise mi")
    # groups = df[1]['Group'].dropna().values.tolist()
    # grp_attr = df[1]["Group-Attr"].dropna().tolist()
    # gr_scores = cal_pairwise_mi(groups, grp_attr, attr_names, mis)
    visualize_for_me(mis, attr_names, {})

    logging.info("drawing heatmap")
    generate_heatmap(mis.copy(), n, attr_names, sorted_attrs, spth)


