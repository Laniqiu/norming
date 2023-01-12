#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
rebuild PPMI via SVD
"""

from __future__ import division
import numpy as np
from scipy import linalg, mat, dot
from scipy.sparse.linalg import svds

from common.setup import logging


def svd(M, dim=300, w=1):
    """

    @param M: the matrix you want to reduce
    @param dim: number of dimensions to which you want to reduce the matrix
                for word embeddings, the typical value is 300
    @param w: weight to assign to the matrix of singular values (1 should be the default one)
    @return:
    """
    logging.info("performing svd...")
    logging.info("settings:")
    logging.info("- dimensions:{}".format(dim))
    logging.info("- weight of singular values:{}".format(dim))

    U, s, Vt = linalg.svd(M)

    U = U[:, 0: dim]
    V = Vt[0: dim, :]

    if not U[U > 0].size > (U.size/2):
        U = -U
        V = -V

    # apply the weighting to the eigenvalues matrix
    s = s**w
    breakpoint()
    # take only the top dimensions of the eigenvalue matrix
    s = s[:dim]
    logging.info("sigma after weighting and truncation:{}".format(s))

    # recombine the matrices
    logging.info("truncated matrix:")
    # rMatrix = dot(U, linalg.diagsvd(s, dim, len(V)))
    rMatrix = np.dot(U, linalg.diagsvd(s, dim, len(V)))
    breakpoint()
    print(rMatrix)



if __name__ == "__main__":
    M = np.array([[0, 59, 4, 0, 39, 23],
                  [6, 52, 4, 26, 58, 4],
                  [33, 115, 42, 17, 83, 10],
                  [9, 12, 2, 27, 17, 3]])
    svd(M, 3, 1)
    # svd_(M, 3)
    