
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "binder"))
import pandas as pd

from binder.mi import main as mi_func
from binder.kmeans import main as kmeans_func
from binder.emb2fea.trainer import main as train
from binder.emb2fea.evaluator import main as evaluate


fpth = "data/meanRating_July1.xlsx"

# mi
spth = "data/out/mi.png"
# mi_func(fpth, spth)

# kmeans
kout = "data/out/kmeans.xlsx"
# kmeans_func(fpth, kout)

# regression
efolder = ""  # the folder to save language model
if not efolder:
    print("no language model given, exit..")
    exit()
tmp_dir = "data/reg_out"
wpth = ""
if not wpth:
    print("no words given, exit..")
    exit()

out_dir = "data/eval"

# train(fpth, efolder, tmp_dir)
evaluate(fpth, tmp_dir, wpth, out_dir)




