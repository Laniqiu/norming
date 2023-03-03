# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 1/3/2023 1:48 pm

"""
from binder.emb2fea.trainer import baseline
from common.setup import adr

_dir = adr.joinpath("binder")

baseline(_dir.joinpath("new_ratings.xlsx"),
         _dir.joinpath("out")
         )
