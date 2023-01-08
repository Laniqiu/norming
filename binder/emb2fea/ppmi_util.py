# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 7/1/2023 9:56 pm

"""
from scipy.sparse import csr_matrix, vstack, save_npz, load_npz, hstack
from scipy.sparse.linalg import svds
import numpy as np
from tqdm import tqdm
import torch

from common.setup import logging, adr
from common.io_utils import general_reader


def svds_by_blocks(epth, out_dir, fpth, chunk=1000,
                   ncols=352275, total=352278, k=300):
    """

    @param epth:
    @param out_dir:
    @param fpth:
    @param chunk:
    @param ncols: col of matrix
    @param vocab_size: row of matrix
    @param total:
    @return:
    """

    n = {3000: 1277, 1000: 277, 2000: 277}.get(chunk, False)
    stpwords = [each.split("\t")[0] for each in general_reader(fpth)][:n]

    if not out_dir.exists():
        out_dir.mkdir()

    logging.info("setting:\n\tchunk={}\n\tstopwords size:"
                 "{}\n\tk={}\n\tout_dir:{}".format(chunk, n, k, out_dir))

    lino, count = -1, 0
    data, row, col = [], [], []
    psbar = tqdm(total=total)
    with open(epth, "r") as fr:
        for line in fr:
            psbar.update(1)
            if lino < 0:
                lino += 1
                logging.info("skip first line ...\n{}".format(line))
                continue

            line = line.strip().split()

            if line[0] in stpwords:
                logging.info("discarding word:{}".format(line[0]))
                continue

            if count == chunk:  # data存储0 ~ k-1共 k 行, 开始生成matrix
                logging.info("lino: {}".format(lino))
                logging.info("generating matrix...")
                mat = csr_matrix((data, (row, col)), shape=(chunk, ncols))
                u, s, vt = svds(mat, k=k)


                # u, s, vt 保存方式不同
                lazy_save(out_dir, "vt", lino, csr_matrix(vt))
                lazy_save(out_dir, "u", lino, csr_matrix(u))
                # s全部保存
                lazy_save(out_dir, "s", lino, csr_matrix(s))

                del u
                del s
                del vt

                data, row, col = [], [], []  # 记载当前行
                count = 0

            this_data, this_row, this_col = process_line(count, line)
            data += this_data
            row += this_row
            col += this_col

            count += 1
            lino += 1
        logging.info("vocab size: {}".format(lino))


def lazy_save(out_dir, fname, lino, u):
    """
    @param out_dir:
    @param fname:
    @param lino:
    @param u:
    @return:
    """

    this_fout = out_dir.joinpath("{}_{}".format(fname, lino))
    logging.info("saving new matrix at {} ...".format(this_fout))
    save_npz(this_fout, u)


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


def save_mat(out_dir, fname, lino, smat, torch_save=False):
    """
    save csr_matrix
    @param out_dir:
    @param fname:
    @param lino:
    @param smat:
    @return:
    """
    fout = out_dir.joinpath("{}_{}".format(fname, lino))
    if torch_save:
        torch.save(smat, fout)
    else:
        save_npz(fout, csr_matrix(smat))


if __name__ == "__main__":
    from pathlib import Path
    _root = Path(adr).joinpath("dough")

    svds_by_blocks(_root.joinpath("embeddings/ppmi.wiki.word"),
                   _root.joinpath("tmp"),  # 保存路径
                   # Path("/content/tmp"),  # 保存路径
                   _root.joinpath("ppmi/filter_1277.txt"),  # 过滤词的路径
                   chunk=1000
                   )
