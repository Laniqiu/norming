# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 5/2/2023 4:58 pm

"""
import sys
sys.path.insert(0, '/Users/laniqiu/Drive/pies/')
sys.path.insert(0, "/home/qiule/drive/pies")
from pprint import pprint
import re

from common.setup import adr
from common.io_utils import general_reader

fin = adr.joinpath("fin/tweetfinsent/raw_test.json")
pat = '"full_text": .+?"is_quote_status"'

with fin.open(mode="r") as fr:
    for line in fr:
        breakpoint()
