# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 2/2/2023 3:09 pm

"""
import sys
sys.path.insert(0, '/Users/laniqiu/Drive/pies/')
sys.path.insert(0, "/home/qiule/drive/pies")


from time import sleep
from random import randint
import stweet as st

from common.setup import logging, adr
from common.io_utils import load_json


def scrape_by_ids(fout, id_str):
    id_task = st.TweetsByIdTask(id_str)
    output_json = st.JsonLineFileRawOutput(fout)

    output_print = st.PrintFirstInBatchRawOutput()

    st.TweetsByIdRunner(tweets_by_id_task=id_task,
                        raw_data_outputs=[output_print, output_json]).run()

_path = adr.joinpath("fin/tweetfinsent/")
fin = _path.joinpath("TweetFinSent_Test.json")
out_dir = _path.joinpath("test")
if not out_dir.exists():
    out_dir.mkdir()

ids = [each.get("Tweet_ID") for each in load_json(fin)]

for id_s in ids:
    fout = out_dir.joinpath("{}.json".format(id_s))
    if fout.exists():
        continue
    logging.info("currently processing {}".format(id_s))
    scrape_by_ids(adr.joinpath(fout), id_s)
    sleep(randint(7, 20))
