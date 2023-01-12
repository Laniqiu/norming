# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 7/1/2023 9:56 pm

"""
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz, hstack
from scipy.sparse.linalg import svds
from scipy import linalg
import numpy as np
from tqdm import tqdm

from common.setup import logging, adr


def bsvds(epth, fout, chunk=1000,
          ncols=352275, total=352278, k=300):
    """

    @param epth:
    @param fout:
    @param fpth:
    @param chunk:
    @param ncols: col of matrix
    @param vocab_size: row of matrix
    @param total:
    @return:
    """

    lino, count = -1, 0
    words, data, row, col = [], [], [], []
    psbar = tqdm(total=total)
    with open(epth, "r") as fr:
        for line in fr:
            psbar.update(1)
            if lino < 0:
                lino += 1
                logging.info("skip first line ...\n{}".format(line))
                continue

            line = line.strip().split()

            if count == chunk:  # data存储0 ~ k-1共 k 行, 开始生成matrix
                logging.info("lino: {}".format(lino))
                mat2values(data, row, col, words, fout, chunk, k, ncols)
                count = 0

            this_data, this_row, this_col = process_line(count, line)
            data += this_data
            row += this_row
            col += this_col
            words.append(line[0])

            count += 1
            lino += 1

    if count > 0:
        logging.info("final ...")
        mat2values(data, row, col, words, fout, chunk, k, ncols)


def mat2values(data, row, col, words, fout, chunk, k, ncols):
    logging.info("generating matrix...")
    mat = csr_matrix((data, (row, col)), shape=(chunk, ncols))
    logging.info("implementing svd ...")
    u, s, vt = svds(mat, k=k)
    sig = csr_matrix(np.eye(k) * s)
    values = np.dot(csr_matrix(u), sig).toarray().astype(str)
    logging.info("writing to file ...")
    with open(fout, "a") as fw:
        for idx, w in enumerate(words):
            this_value = " ".join(values[idx].tolist())
            outt = "{} {}\n".format(w, this_value)
            fw.write(outt)

    words.clear()
    data.clear()
    row.clear()
    col.clear()


def process_line(count, line):
    this_data, this_row, this_col = [], [], []
    for each in line[1:]:
        res = each.split(":")
        if len(res) != 2:
            continue
        _id, _va = res
        this_data.append(float(_va))
        this_row.append(count)
        this_col.append(int(_id))
    return this_data, this_row, this_col


def sim_test(mpth, test_words, topn=10, total=352277):
    """
    test embedding via cos sim
    @param mpth:
    @return:
    """
    from sklearn.metrics.pairwise import cosine_similarity
    from pprint import pprint

    psbar = tqdm(total=total)
    def get_vecs():
        vecs = {}
        with open(mpth, "r") as fr:
            psbar.update(1)
            for line in fr:
                line = line.strip().split()
                if len(vecs) == len(test_words):
                    return vecs
                if line[0] in test_words:
                    vecs[line[0]] = np.array([float(i) for i in line[1:]]).reshape(1, -1)
        return vecs

    vecs = get_vecs()
    show = []
    for w, this_vec in vecs.items():
        for that_w, that_vec in vecs.items():
            if that_w == w:
                continue
            s_ = cosine_similarity(that_vec, this_vec)[0][0]
            show.append((w, that_w, s_))
    pprint(show)
    # with open(mpth, "r") as fr:
    #     for line in fr:
    #         psbar.update(1)
    #         line = line.strip().split()
    #         that_vec = np.array([float(i) for i in line[1:]]).reshape(1, -1)
    #         sim = cosine_similarity(that_vec, this_vec)[0][0]
    #         score.append((line[0], sim))
    #         if len(score) > topn:  #
    #             score.sort(key=lambda x: x[1], reverse=False)  # 从大到小
    #             score = score[:topn]
    # pprint(score)




if __name__ == "__main__":
    from pathlib import Path

    _root = Path(adr).joinpath("dough")

    # bsvds(_root.joinpath("embeddings/ppmi.wiki.word"),
    #       # _root.joinpath("tmp"),  # 保存路径
    #       _root.joinpath("ppmi/rebuild.ppmi.wiki.word"),
    #       chunk=1000
    #       )

    sim_test(_root.joinpath("ppmi/rebuild.ppmi.wiki.word"), ["乳白色", "灰白色", "银白色", "黄白色", "白色"])

