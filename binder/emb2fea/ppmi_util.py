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
import shutil

from common.setup import logging, get_root
from common.io_utils import general_reader


def svds_by_blocks(epth, out_dir, fpth, k=1000,
                   ncols=352275, total=352278,
                   using_torch=False):
    """

    @param epth:
    @param out_dir:
    @param fpth:
    @param k:
    @param ncols: col of matrix
    @param vocab_size: row of matrix
    @param total:
    @return:
    """

    n = {3000: 1277, 1000: 277, 2000: 277}.get(k, False)
    stpwords = [each.split("\t")[0] for each in general_reader(fpth)][:n]

    logging.info("setting:\n\tk={}\n\tstopwords size:{}".format(k, n))

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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

            if count == k:  # data存储0 ~ k-1共 k 行, 开始生成matrix
                logging.info("lino: {}".format(lino))
                logging.info("generating matrix...")
                # if using_torch:
                #     mat = torch.sparse_csr_tensor(torch.tensor(row, dtype=torch.int64),
                #                                   torch.tensor(col, dtype=torch.int64),
                #                                   torch.tensor(data), dtype=torch.float
                #                                   ).to(device)
                #     u, s, vt = torch.svd_lowrank(mat, q=k-1)
                # else:
                mat = csr_matrix((data, (row, col)), shape=(k, ncols))
                u, s, vt = svds(mat, k=k - 1)

                lazy_save(out_dir, "u", lino, u)
                lazy_save(out_dir, "s", lino, s)
                lazy_save(out_dir, "vt", lino, vt)

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
    that_fout = out_dir.joinpath("{}_{}.npz".format(fname, lino - 1000))
    this_fout = out_dir.joinpath("{}_{}".format(fname, lino))

    if that_fout.exists():  # 若上一个存在，则和之前的合并保存，然后删除原来的
        this_u = vstack((load_npz(that_fout), u))
        logging.info("saving stacked matrix ...")
        save_npz(this_fout, this_u)
        that_fout.unlink(missing_ok=True)
    else:
        logging.info("saving new matrix...")
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
    _root = Path(get_root()).joinpath("dough")

    svds_by_blocks(_root.joinpath("embeddings/ppmi.wiki.word"),
                  # _root.joinpath("ttmp")
                   Path("/content/tmp"),  # 保存路径
                   _root.joinpath("ppmi/filter_1277.txt"),  # 过滤词的路径
                   k=3000,
                   using_torch=False
                  )
