# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 26/2/2023 2:23 pm
提取单义项target word的sense emb
"""
import sys
sys.path.insert(0, "/home/qiule/pies/")

from common.setup import adr, logging
from common.io_utils import load_json, dump_json
from utils import load_models, collect_embs, save_vecs

def main(fin, model_path, temp_dir):
    """

    @param fin:
    @param model_path: str
    @param temp_dir:
    @return:
    """
    if not temp_dir.exists():
        temp_dir.mkdir(exist_ok=True)

    logging.info("loading data from {}...".format(fin.name))
    data = load_json(fin)

    logging.info("loading model from {}".format(model_path.name))
    model, tokenizer = load_models(str(model_path))
    for tgt, sents in data.items():
        sents_vec = collect_embs(tgt, sents, model, tokenizer)
        sense_vec = sents_vec.mean(axis=0).unsqueeze(0)
        fout = temp_dir.joinpath(tgt)
        save_vecs(sense_vec, fout)

if __name__ == "__main__":
    _dir = adr.joinpath("Jing")

    main(_dir.joinpath("samps/100_samps.json"),   # 输入json文件
         adr.joinpath("lfs/chinese_roberta_wwm_ext_pytorch"),  # lm模型路径
         _dir.joinpath("mono_sens")  # 单义词sense emb保存目录
    )






