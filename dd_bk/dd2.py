"""
parsing only
"""

from common.io_utils import read_csv_pd, general_reader, read_xlsx_pd
import stanza
from stanza.models.common.doc import Document
import numpy as np
import pandas as pd
from collections import OrderedDict
import re
import codecs
import json
from dependency_distance import get_pos_map
import logging

def find_roots(doc):
    ids, ws = [], []
    for _, sent in enumerate(doc.sentences):
        for word in sent.words:
            if word.head == 0:
                ids.append(word.id)
                ws.append(word.text)
    return ids, ws

def depth2root(roots, doc, wids):
    """
    """
    diss, details = [], []
    for s_id, sent in enumerate(doc.sentences):
        d, ws = [], []
        root = roots[s_id]
        word_ids = wids[s_id]
        for word in sent.words:
            if word.head == 0:  # excluding root
                ws.append("root")
                d.append(0)
                continue
            if word_ids[word.id-1] == "*":  # excluding puncts
                continue
            ws.append(word.text)  # 需要处理b
            if word.head == root:
                d.append(1)
                continue
            arc = 1
            while word.head and word.head != roots[s_id]:  # 必须确定有root
                word = sent.words[word.head - 1]
                arc += 1
            d.append(arc)
        assert  len(ws) == len(d)
        zipped = list(zip(ws, d))
        details.append(zipped)
        diss.append(np.round(np.array(d).mean(), 2))
    # return np.array(diss).reshape((-1, 1)), np.array(details).reshape((-1, 1))
    return np.array(diss).reshape((-1, 1)), details

def ldd2root(roots, doc, wids):
    """
    linear dependeny distance to root
    excluding the root
    :param doc:
    :return:
    """
    to_root, details = [], []
    for idx, sent in enumerate(doc.sentences):
        word_ids = wids[idx]
        dis, ws = [], []
        for word in sent.words:  #
            if word.head == 0:  # excluding the root
                ws.append("root")
                dis.append(0)
                continue
            if word_ids[word.id - 1] == "*":
                continue
            count = 0
            start, end = min(word.id, roots[idx])-1, max(word.id, roots[idx])
            for s in word_ids[start: end]:
                if s == "*":
                    count += 1
            ws.append(word.text)
            dis.append(abs(word.id - roots[idx] - count))
        assert len(ws) == len(dis)
        zipped = list(zip(ws, dis))
        details.append(zipped)
        to_root.append(np.round(np.array(dis).mean(), 2))

    to_root = np.array(to_root).reshape((-1, 1))
    # return to_root, np.array(details).reshape((-1, 1))
    return to_root, details

def ldd2head(doc, wids):
    """
    excluding the root
    :param doc:
    :return:
    """
    diss, details = [], []
    for _, sent in enumerate(doc.sentences):
        dis, ws = [], []
        word_ids = wids[_]
        # 有错， 要改
        for word in sent.words:
            if word.head == 0:
                ws.append("root")
                dis.append(0)
                continue
            if word_ids[word.id-1] == "*":
                continue
            count = 0
            start, end = min(word.id, word.head) - 1, max(word.id, word.head)
            for s in word_ids[start: end]:
                if s == "*":
                    count += 1
            dis.append(abs(word.id - word.head - count))
            ws.append(word.text)
        assert len(ws) == len(dis)
        zipped = list(zip(ws, dis))
        details.append(zipped)
        diss.append(np.round(np.array(dis).mean(), 2))
    # return np.array(diss).reshape((-1, 1)), np.array(details).reshape((-1, 1))
    return np.array(diss).reshape((-1, 1)), details

def load_sents(fpth):
    """
    load sents from xlsx (from boh) and remove duplications
    :param fpth:
    :return:
    """
    def remove_u200b(ii):
        return str(ii).replace(u"\u200b", "")

    pd_data = read_xlsx_pd(fpth, sheet_name=1, cols=["SENTENCE", "WORD", "IA_LABEL"])
    # remove duplicated sents
    for _, num in enumerate(pd_data["WORD"]):
        if num in pd_data["WORD"][:_]:
            pd_data["WORD"][_] = np.nan
    pd_data.dropna(axis=0, subset=["WORD"], inplace=True)
    data = pd_data.values
    sents = []
    for i in np.arange(data[:, 0].min(), data[:, 0].max() + 1):
        idx = np.where(data[:, 0] == i)
        sent = data[:, 2][idx].tolist()
        sent = list(map(remove_u200b, sent))
        sents.append(sent)
    logging.info("sentence count **:{}".format(len(sents)))
    return sents


def load_sents_parts(fpth, cols=["SENTENCE", "WORD", "IA_LABEL", "POS"]):
    """
    :param fpth:
    :return: dict of sents, sent id: sent
    """
    def remove_u200b(ii):
        return str(ii).replace(u"\u200b", "")

    all_ = pd.read_table(fpth, sep="\t")
    data = all_[cols].values
    sents = OrderedDict()
    # for i in np.arange(data[:, 0].min(), data[:, 0].max() + 1):
    for each in data:
        sid = each[0]
        idx = np.where(data[:, 0] == sid)
        wid = data[:, 1][idx].tolist()
        sent = data[:, 2][idx].tolist()
        sent = list(map(remove_u200b, sent))
        assert len(wid) == len(sent)
        sents[sid] = list(zip(wid, sent))
    logging.info("sentence count **:{}".format(len(sents)))
    return sents, all_

def pos_tag_canto(sents, upos_map=None):
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
        segged[sid] = (rr, wid)
    return segged

def pos_tag_mandarin_jiagu(sents, upos_map):
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

        poses = [upos_map[p] for p in jiagu.pos(words)]
        rr = list(zip(words, poses))

        segged[sid] = (rr, wid)
    return segged

def d_parsing(segged, lang="zh-hant"):
    nlp = stanza.Pipeline(lang=lang, processors='depparse', depparse_pretagged=True)
    sents = []
    # sent_id = []
    word_id = []
    for s_id, (sen_ps, wid) in segged.items():
        each_sen = []
        # sent_id.append(s_id)
        word_id.append(wid)
        # print(sen_ps)
        for idx, (w, p) in enumerate(sen_ps):
            # todo 简易版，再改
            each_w =  {'id': idx + 1, 'text': w, 'lemma': w, 'upos': p}
            each_sen.append(each_w)
        sents.append(each_sen)
    sents = Document(sents)
    doc = nlp(sents)

    # cal dd, 去除标点符号
    rt_id, rt_txt = find_roots(doc)
    roots = list(zip(rt_txt, rt_id))
    roots = list(map(str, roots))

    assert len(doc.sentences) == len(rt_id)

    dep2root, dwp = depth2root(rt_id, doc, word_id)
    dis2root, dwr = ldd2root(rt_id, doc, word_id)
    dis2head, dwh = ldd2head(doc, word_id)

    return roots, dis2root, dwr, dis2head, dwh, dep2root, dwp

    # output = np.concatenate((np.array(roots).reshape((-1, 1)), dis2root, dwr, dis2head, dwh, dep2root, dwp), axis=1)
    # return output
    # ouput = np.concatenate((np.array(sent_id).reshape((-1, 1)), np.array(roots).reshape((-1, 1)),
    #                         dis2root, dwr, dis2head, dwh, dep2root, dwp), axis=1).tolist()

    # fw = open(fout, encoding="utf-8-sig", mode="w")
    # writer = csv.writer(fw)
    # header = ["sentence", "root", "linear dd to root", "details",  \
    #            "linear dd to head", "details", "depth to root", "details"]
    # writer.writerow(header)
    # writer.writerows(ouput)
    # fw.close()



if __name__ == "__main__":
    import os
    root = "/mnt/c/Users/22068051r/OneDrive - The Hong Kong Polytechnic University/" \
           "assignments/dependency_distance/boh/"


    from glob import glob

    mpth = os.path.join(root, "upos_map.txt")
    upos_map = get_pos_map(mpth)

    files = glob(os.path.join(root, "annotator_avg/*.txt"))
    for f in files:
        if not f.endswith("tsr_281.txt"):
            continue
            # lang= "zh"
            # pos_func = pos_tag_mandarin_jiagu
        # else:

        lang = "zh-hant"
        pos_func = pos_tag_canto
        logging.info("language:", lang)
        fout = f.replace("annotator_avg", "res")
        sents, all_ = load_sents_parts(f)
        segged = pos_func(sents)


        roots, dis2root, dwr, dis2head, dwh, dep2root, dwp = d_parsing(segged, lang=lang)

        dwri = [e[1] for l in dwr for e in l]
        dwhi = [e[1] for l in dwh for e in l]
        dwpi = [e[1] for l in dwp for e in l]
        # print(dwri.pop(883))
        # print(dwhi.pop(883))
        # print(dwpi.pop(883))

        # 在all_后面加四列，
        all_["LD2ROOT"] = dwri
        all_["LD2HEAD"] = dwhi
        all_["DEPTH2ROOT"] =  dwpi
        all_.to_csv(fout, sep="\t")



    # fpth = os.path.join(root, "processed/simp/NR/part_0/LP1_05.txt")  # part 0
    # sents = load_sents_parts(fpth)
    # segged = pos_tag_canto(sents)
    # mpth = os.path.join(root, "upos_map.txt")
    # "simp_1_all_noRep.xlsx", "trad_1_all_noRep.xlsx"
    # upos_map = get_pos_map(mpth)
    # segged = pos_tag_mandarin_jiagu(sents)
    # todo 保存分词结果 后面再说
    # d_parsing(segged, fpth.replace(".txt", ".csv"), lang="zh")



