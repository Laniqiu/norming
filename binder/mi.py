
import pandas as pd

from common.setup import logging

from mi_replication import cal_pairwise_mi, cal_bt_attrs, \
    generate_heatmap, visualize_for_me


def main(fpth, spth, handle_nans="pad"):
    """
    @param ratings: pandas series
    @param attr_names:  list
    @param sorted_attrs: list
    @param spth: path to save the heatmap
    @param handle_nans: how to deal with nan values
    @return:
    """
    ratings, attr_names, sorted_attrs = load_data(fpth)
    n = ratings.shape[1]
    logging.info("compute pairwise mi")
    mis = cal_bt_attrs(n, ratings, method=handle_nans)
    logging.info("compute in-group pairwise mi")
    visualize_for_me(mis, attr_names, {})

    logging.info("drawing heatmap")
    generate_heatmap(mis.copy(), n, attr_names, sorted_attrs, spth)


def load_data(fpth):
    df = pd.read_excel(fpth)
    ratings = df.iloc[5:, 11:]
    attr_names = df.iloc[3, 11:].to_list()
    sorted_attrs = attr_names
    return ratings, attr_names, sorted_attrs


if __name__ == '__main__':
    fpth = "../data/meanRating_July1.xlsx"
    spth = "../data/out/mi.png"

    main(fpth, spth)

