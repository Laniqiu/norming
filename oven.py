# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 1/3/2023 1:48 pm

"""
# import os
# os.environ["LOG_FILE"] = "logs/log.log"
# os.environ["LOG_IN_FILE"] = "0"


# from binder.emb2fea.trainer import main as train
# from common.setup import adr
# from Jing.extract_sense_emb import collect

from common.setup import adr
from binder.emb2fea.trainer import main


efolder = adr.joinpath("embeddings")
fpth = adr.joinpath("new_ratings.xlsx")
out_dir = adr.joinpath("out_grouping")

main(fpth, efolder, out_dir)