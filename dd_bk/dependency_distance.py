# -*- coding:utf-8 -*-
"""
calculate dependency distance for each word to the head
"""

from common.util import read_csv_pd, general_reader, read_xlsx_pd
import stanza
from stanza.models.common.doc import Document
import numpy as np
import csv


def main(fpth, fout, lang="zh-hans"):
    data = read_csv_pd(fpth).values

    nlp = stanza.Pipeline(lang=lang, processors='depparse', depparse_pretagged=True)
    if lang == "zh-hans":
        from zhconv import convert

    sents = []
    sl = []
    for row in data:
        _, sen, tk, tk_pt, n, sen_ps = row
        # n： excluding punctuations base don the pos
        sen_ps = [(w, p) for (w, p) in eval(sen_ps) if p.isalpha() and p.strip() != "w"]
        """pretagged"""
        each_sen = []
        for idx, (w, p) in enumerate(sen_ps):
            # 需要处理
            if not p.isalpha():  #
               p = "w"
            upos = upos_map[p]
            xpos = ""
            feat = feat_map(upos, w) if lang == "zh-hans" else feat_map(upos, convert(w, "zh-hant"))
            each_w =  {'id': idx + 1, 'text': w, 'lemma': w, 'upos': upos_map[p], 'xpos': xpos, 'feat':"|".join(feat)}
            each_sen.append(each_w)
        sents.append(each_sen)
        sl.append(idx+1)
    sents = Document(sents)
    doc = nlp(sents)
    # print(*[
    #     f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}'
    #     for sent in doc.sentences for word in sent.words], sep='\n')

    # cal dependency distance  (average distance of each word to the root)
    rt_id, rt_txt = find_roots(doc)
    roots = list(zip(rt_txt, rt_id))
    roots = list(map(str, roots))

    assert len(doc.sentences) == len(rt_id)

    dep2root, dw1 = depth2root(rt_id, doc)
    dis2root, dw2 = ldd2root(rt_id, doc)
    dis2head, dw3 = ldd2head(doc)

    output = np.concatenate((data, np.array(sl).reshape(-1,1), np.array(roots).reshape((-1, 1)), dis2root, dw2,
                             dis2head, dw3, dep2root, dw1), axis=1).tolist()
    fw = open(fout, encoding="utf-8-sig", mode="w")
    writer = csv.writer(fw)
    header =  ["index", "sentences", "tokens", "tokens_with_puncht", \
               "n_tokens", "postag", "sent length", "root", "linear dd to root", "details",  \
               "linear dd to head", "details", "depth to root", "details"]
    writer.writerow(header)
    writer.writerows(output)

    fw.close()

def find_roots(doc):
        ids, ws = [], []
        for sent in doc.sentences:
            for word in sent.words:
                if word.head == 0:
                    ids.append(word.id)
                    ws.append(word.text)
        return ids, ws

def depth2root(roots, doc):
    """
    """
    diss, details = [], []
    for s_id, sent in enumerate(doc.sentences):
        d, ws = [], []
        print("** looping sentence {}".format(s_id))
        root = roots[s_id]
        for word in sent.words:
            # word = sent.words[w_id]
            # print("word:", word.text)
            if word.head == 0:  # excluding root
                continue
            ws.append(word.text)
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
    return np.array(diss).reshape((-1, 1)), np.array(details).reshape((-1, 1))

def ldd2root(roots, doc, exclude_root=True, exclude_punct=False):
    """
    linear dependeny distance to root
    excluding the root
    :param doc:
    :return:
    """
    to_root, details = [], []
    for idx, sent in enumerate(doc.sentences):
        dis, ws = [], []
        for word in sent.words:  # including puncts
            # !! excluding the root??
            if word.head == 0:
                continue
            ws.append(word.text)
            dis.append(abs(word.id - roots[idx]))
        assert len(ws) == len(dis)
        zipped = list(zip(ws, dis))
        details.append(zipped)
        to_root.append(np.round(np.array(dis).mean(), 2))

    to_root = np.array(to_root).reshape((-1, 1))
    return to_root, np.array(details).reshape((-1, 1))

def ldd2head(doc):
    """
    excluding the root
    :param doc:
    :return:
    """
    diss, details = [], []
    for sent in doc.sentences:
        dis, ws = [], []
        for word in sent.words:
            if word.head == 0:
                continue
            dis.append(abs(word.id - word.head))
            ws.append(word.text)
        assert len(ws) == len(dis)
        zipped = list(zip(ws, dis))
        details.append(zipped)
        diss.append(np.round(np.array(dis).mean(), 2))
    return np.array(diss).reshape((-1, 1)), np.array(details).reshape((-1, 1))

def feat_map(upos, w):
    """traditional chinese feats"""
    feat = []
    if upos in ["NOUN", "PART", "PRON"] and w[-1] in ["們", "们"]:
        feat.append("Number=Plur")
    if upos in ["ADP"] and w in ["之外"]:
        feat.append("Case=Gen")
    if upos in ["PART"]:
        if w in ["的", "之", "地"]:
            feat.append("Case=gen")
        if w in ["呢", "嗎", "啊"]:
            feat.append("PartType=Int")
    if upos in ["ADV"] and w in ["不", "未", "沒", "別", "無"]:
        feat.append("Polarity=Neg")
    if upos in ["AUX", "PART"] and w in ["了", "過"]:
        feat.append("Aspect=Perf")
    if upos in ["AUX"] and w in ["著"]:
        feat.append("Aspect=Prog")
    if upos in ["ADP", "VERB"] and \
            w in ["以", "使", "讓", "使得", "令", "導致", "要求", "派", "派遣", "任命"]:
        feat.append("Voice=Cau")
    if upos in ["AUX", "VERB"] and w in ["被", "為"]:
        feat.append("Voice=Pass")
    # if upos in ["NUM"]:

    return feat

def get_pos_map(mpth):
    map_d = {}
    for line in general_reader(mpth)[1:]:
        n, _, pos = line.strip().split("\t")
        map_d[n] = pos
    return map_d


if __name__ == "__main__":

    import os
    root = "/mnt/c/Users/22068051r/OneDrive - The Hong Kong Polytechnic University/"
    tra_file = os.path.join(root, "assignments/dependency_distance/sentences_canto_debugged.csv")
    sim_file = os.path.join(root, "assignments/dependency_distance/sentences_mandarin_debugged.csv")

    pos1, pos2 = os.path.join(root, "pies/dependency/pos_canto.txt"), \
                 os.path.join(root, "pies/dependency/pos_mandarin.txt")
    # uni_poses(pos1, pos2)
    mpth = os.path.join(root, "assignments/dependency_distance/upos_map.txt")
    # "simp_1_all_noRep.xlsx", "trad_1_all_noRep.xlsx"
    upos_map = get_pos_map(mpth)
    # zh-hans:简体，zh-hant繁体
    # main(tra_file, "canto.csv", lang="zh-hant")
    # main(sim_file, "mandarin.csv", lang="zh-hans")

    # read_xlsx_pd(os.path.join(root, "assignments/dependency_distance/simp_1_all_noRep.xlsx"), cols=[])









