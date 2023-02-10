# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 10/2/2023 2:53 pm

"""
import sys
import pandas as pd

sys.path.insert(0, '/Users/laniqiu/Drive/pies/')
sys.path.insert(0, "/home/qiule/drive/pies")



from common.setup import adr

if __name__ == "__main__":

    fpth = adr.joinpath("binder/new_ratings.xlsx")
    data = pd.read_excel(fpth)
    breakpoint()

