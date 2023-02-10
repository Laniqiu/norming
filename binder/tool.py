# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 7/2/2023 10:36 am

"""
import sys
sys.path.insert(0, '/Users/laniqiu/Drive/pies/')
sys.path.insert(0, "/home/qiule/drive/pies")

import pandas as pd

from common.setup import adr

_path = adr.joinpath("binder/new4repeated_words")

files = _path.glob("*.xlsx")

for f in files:
    print(f.name)
    each = pd.read_excel(f)
    print(each.iloc[:, 68].name)
    print(each.iloc[:, 69].name)
    print(each.iloc[:, 70].name)
breakpoint()


if __name__ == "__main__":
    pass