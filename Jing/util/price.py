# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 25/2/2023 4:35 pm
计算人工费
"""
from os import environ
environ["LOG_IN_FILE"] = "0"

import pandas as pd
import sys
sys.path.insert(0, "/Users/laniqiu/Kitchen/pies/")


from common.setup import adr

_dir = adr.joinpath("Jing")

file = _dir.joinpath("Copy of 青青high_75.xlsx")

# file = _dir.joinpath(sys.argv[1])

count = 0
all_shts = pd.read_excel(file, sheet_name=None)
for name, sht in all_shts.items():
    # 如果当前义项的exp数目为0，去除
    nrow = sht.shape[0]
    count += nrow
    for i in range(nrow):
        if sht.iloc[i].isna().sum() >= 10:
            count -= 1
            continue

        if name.strip() not in sht.iloc[i, 1]:

            print(name)
            count -= 1

print("{}:".format(file.stem))
print("{} * {}= {}".format(count, 2, count*2))


if __name__ == "__main__":
    pass
