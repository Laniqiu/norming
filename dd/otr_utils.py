
import re
from collections import OrderedDict
import pandas as pd
import numpy as np
import logging

from common.io_utils import general_reader

logging.basicConfig(level=logging.INFO)

def get_pos_map(mpth):
    """
    记载词性映射字典
    """
    map_d = {}
    for line in general_reader(mpth)[1:]:
        n, _, pos = line.strip().split("\t")
        map_d[n] = pos
    return map_d


def load_sents_parts(fpth, cols=["SENTENCE", "WORD", "IA_LABEL", "POS"]):
    """
    加载数据，返回指定列和全部数据
    :param fpth:
    :return: dict of sents, sent id: sent
    """
    def remove_u200b(ii):
        return str(ii).replace(u"\u200b", "")

    all_ = pd.read_table(fpth, sep="\t")
    data = all_[cols].values
    sents = OrderedDict()
    sids = set(data[:, 0].tolist())
    for sid in sids:
        idx = np.where(data[:, 0] == sid)
        wid = data[:, 1][idx].tolist()
        sent = data[:, 2][idx].tolist()
        sent = list(map(remove_u200b, sent))
        assert len(wid) == len(sent)
        sents[sid] = list(zip(wid, sent))
    logging.info("sentence count **:{}".format(len(sents)))
    return sents, all_



def pos_tag_canto(sents):
    """
    粤语词性标注（已分词）
    :param sents:
    :return:
    """
    import pycantonese as tagger
    segged = OrderedDict()
    pat1 = "(\W+)(\w+)"  # 标点只在开头或结尾
    pat2= "(\w+)(\W+)"
    for sid, sent in sents.items():
        wid, words = [], []
        for id, w in sent:
            # 标点需要处理，以及id
            w = w.strip().replace(" ", "")
            mm1 = re.search(pat1, w)
            mm2 = re.search(pat2, w)

            if mm1:
                punct = mm1.groups()[0]
                for p in punct:
                    wid.append("*")
                    words.append(p)

                w = w[len(punct):]

            stars_id, stars = [], []
            if mm2:
                punct = mm2.groups()[1]
                for p in punct:
                    stars_id.append("*")
                    stars.append(p)

                w = w[:len(w)-len(punct)]
            wid.append(id)
            words.append(w)
            wid += stars_id
            words += stars
            # 得把标点分开
        rr = tagger.pos_tag(words)
        assert len(rr) == len(wid)
        _, poses = list(zip(*rr))
        segged[sid] = list(zip(wid, words, poses))
    return segged

def pos_tag_mandarin_jiagu(sents):
    """
    简体词性标注（已分词）
    """
    import jiagu
    segged = OrderedDict()
    pat1 = "(\W+)(\w+)"  # 标点只在开头或结尾
    pat2 = "(\w+)(\W+)"

    for sid, sent in sents.items():
        wid, words = [], []
        for id, w in sent:
            # 标点需要处理，以及id
            w = w.strip().replace(" ", "")
            mm1 = re.search(pat1, w)
            mm2 = re.search(pat2, w)
            if mm1:
                punct = mm1.groups()[0]
                for p in punct:
                    wid.append("*")
                    words.append(p)

                w = w[len(punct):]

            stars_id, stars = [], []
            if mm2:
                punct = mm2.groups()[1]
                for p in punct:
                    stars_id.append("*")
                    stars.append(p)

                w = w[:len(w)-len(punct)]

            wid.append(id)
            words.append(w)
            wid += stars_id
            words += stars
        poses = jiagu.pos(words)
        assert len(poses) == len(wid)
        segged[sid] = list(zip(wid, words, poses))
    return segged