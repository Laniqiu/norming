
import os
import sys
sys.path.insert(0, os.path.join(os.getcwd(), "binder"))


from binder.mi import main as mi_func
from binder.kmeans import main as kmeans_func
from binder.emb2fea.trainer import main as train
from binder.emb2fea.evaluator import main as evaluate


fpth = "data/new_ratings.xlsx"

# mi
sv_dir = "data/out/"
mi_func(fpth, sv_dir)

# kmeans
kout = "data/out/kmeans.xlsx"
kmeans_func(fpth, kout)

# regression
# efolder = ""  # !! this is the folder to save language model
efolder = "/disk/lani/dough/lfs"
tmp_dir = "data/reg_out"

if not efolder:
    print("no language model given, exit..")
    exit()

out_dir = "data/out/eval"
train(fpth, efolder, tmp_dir)
evaluate(fpth, tmp_dir, out_dir)




