# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 7/1/2023 7:38 pm
"""
# from scipy.sparse import csr_matrix, vstack, save_npz, load_npz
# from scipy.sparse.linalg import svds

# import imp
# import common
# imp.reload(common)
from common.setup import logging, get_root
from common.io_utils import general_reader, general_writer


def check(fpth, fout, chunk=5000):
    cache = []
    count = 0
    lineno = -1
    with open(fpth, "r") as fr:
        for line in fr:
            if lineno < 0:
                lineno += 1
                continue
            print(line)
            word = line.strip().split()[0]
            cache.append(word + "\n")
            count += 1
            if count == chunk:
                with open(fout, "a") as fw:
                    fw.writelines(cache)
                cache.clear()
                count = 0
    with open(fout, "a") as fw:
        fw.writelines(cache)
        cache.clear()
        count = 0


if __name__ == "__main__":
    from pathlib import Path
    _root = Path(get_root()).joinpath("dough")

    # check(_root.joinpath("embeddings/wiki.zh.vec"),
    #       _root.joinpath("ppmi/wiki.zh.words")
    #       )
