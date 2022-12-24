import logging
import pandas as pd
import numpy as np
import stanza
from stanza.models.common.doc import Document

logging.basicConfig(level=logging.INFO)

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
                ws.append(word.text)
                d.append(-1)
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
        assert len(ws) == len(d)
        zipped = list(zip(ws, d))
        details.append(zipped)
        ad = np.array(d)
        diss.append(np.round(ad[np.where(ad != -1)].mean(), 2))
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
            if word_ids[word.id - 1] == "*":  # punct, ignore
                ws.append(word.text)  # todo
                dis.append(-1)
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
        adis = np.array(dis)
        mdis = np.round(adis[np.where(adis != -1)].mean(), 2)
        to_root.append(mdis)

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
                ws.append(word.text)  # todo
                dis.append(-1)
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
        adis = np.array(dis)
        mdis = np.round(adis[np.where(adis != -1)].mean(), 2)
        diss.append(mdis)
    # return np.array(diss).reshape((-1, 1)), np.array(details).reshape((-1, 1))
    return np.array(diss).reshape((-1, 1)), details

def parsing_dd(files, cols=["sid", "wid", "text", "upos*", "xpos"]):
    """
    parsing and cal dd and save to original files
    file in files: posed texts with sent id, word id, etc.
    """
    nlp_tra = stanza.Pipeline(lang="zh-hant",
                              processors='depparse', depparse_pretagged=True)
    nlp_sim = stanza.Pipeline(lang="zh-hans",
                              processors='depparse', depparse_pretagged=True)

    for f in files:
        logging.info("Processing: {}".format(f.name))
        if "simp" in f.name:
            lang = "zh-hans"
            nlp = nlp_sim
        else:
            lang = "zh-hant"
            nlp = nlp_tra

        logging.info("lang:{}".format(lang))

        all_ = pd.read_table(f, sep="\t")
        data = all_[cols].values

        sents, word_id = [], []
        sent_id = list(set(data[:, 0].tolist()))
        sent_id.sort()
        for sid in sent_id:  # sent id
            each_sen = []
            ixs = np.where(data[:, 0] == sid)
            for idx, (w, p, x) in enumerate(data[:, 2:][ixs]):
                each_w = {'id': idx + 1, 'text': w, 'lemma': w, 'upos': p, "xpos": x}
                each_sen.append(each_w)
            word_id.append(data[:, 1][ixs].tolist())  # word id
            sents.append(each_sen)

        doc = nlp(Document(sents))
        logging.info("Done parsing")
        logging.info("Start calculating dd")

        # cal dd, 去除标点符号
        rt_id, rt_txt = find_roots(doc)
        roots = list(zip(rt_txt, rt_id))
        roots = list(map(str, roots))

        assert len(doc.sentences) == len(rt_id)  # 确保每一棵树都有root

        dep2root, dwp = depth2root(rt_id, doc, word_id)
        dis2root, dwr = ldd2root(rt_id, doc, word_id)
        dis2head, dwh = ldd2head(doc, word_id)

        dwri = [e[1] for l in dwr for e in l]
        dwhi = [e[1] for l in dwh for e in l]
        dwpi = [e[1] for l in dwp for e in l]

        # 规范格式输出  save to files
        all_["LD2ROOT"] = dwri
        all_["LD2HEAD"] = dwhi
        all_["DEPTH2ROOT"] = dwpi
        all_.to_csv(f, sep="\t")


if __name__ == "__main__":
    pass