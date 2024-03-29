
import os
import sys
sys.path.insert(0, os.getcwd())

from binder.emb2fea.trainer import train, baseline
from binder.emb2fea.evaluator import evaluate


def main():
    fpth = "data/ratings.xlsx"
    efolder = "data/embs"

    out_dir = "data/out"
    tmp_dir = "data/out/tmp"
    randm_dir = "data/out/random"

    train(fpth, efolder, tmp_dir)
    baseline(fpth, randm_dir)
    evaluate(fpth, tmp_dir, out_dir)


main()



