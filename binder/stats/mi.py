"""
mutual information
"""
import pandas as pd


from common.setup import *

from mi_replication import cal_bt_attrs, \
    generate_heatmap, visualize_for_me


def main(fpth, out_dir, handle_nans="pad"):
    """
    @param fpth: path the rating file
    @param spth: path to save the heatmap
    @param handle_nans: how to deal with nan values
    @return:
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    df = pd.read_excel(fpth)
    ordered_attrs = [
    "Vision", "Bright", "Dark", "Complexity", "Color", "Pattern", "Shape", "Smooth", "Rough", "Light", "Heavy",
    "Touch", "Small", "Near", "Practice", "UpperLimb", "Hot", "Cold", "Taste", "Smell", "Needs", "Benefit", "Pleasant",
     "Happy", "Drive", "Social", "Communication", "Cognition", "Self", "Arousal", "Attention", "Surprised", "Angry",
     "Sad", "Disgusted", "Unpleasant", "Fearful", "Harm", "Pain", "Caused", "Consequential", "Short", "Long",
     "Duration", "Time", "Loud", "Low", "High", "Music", "Sound", "Audition", "Head", "Speech", "Face", "Human",
     "Body", "Biomotion", "LowerLimb", "Motion", "Fast", "Slow", "Away", "Toward", "Path", "Scene", "Landmark",
     "Large", "Number"
    ]
    new_cols = df.columns[:11].tolist() + [df.iloc[3, 11:].to_list().index(e) + 1 for e in ordered_attrs]
    df = df[new_cols]

    attr_names = df.iloc[3, 11:].to_list()
    sorted_attrs = attr_names

    ratings = df.iloc[5:, 11:]

    n = ratings.shape[1]
    logging.info("compute pairwise mi")
    mis = cal_bt_attrs(n, ratings, method=handle_nans)
    logging.info("compute in-group pairwise mi")
    # groups = df[1]['Group'].dropna().values.tolist()
    # grp_attr = df[1]["Group-Attr"].dropna().tolist()
    # gr_scores = cal_pairwise_mi(groups, grp_attr, attr_names, mis)
    fout = out_dir.joinpath("mi.txt")
    visualize_for_me(mis, attr_names, {}, fout)

    # logging.info("drawing heatmap")
    # spth = out_dir.joinpath("mi_heatmap.png")
    # generate_heatmap(mis.copy(), n, attr_names, sorted_attrs, spth)


if __name__ == "__main__":
    main(
        fpth="ratings_update_Feb24.xlsx",
        out_dir="",

    )