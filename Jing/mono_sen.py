# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 26/2/2023 2:23 pm
提取单义项target word的sense emb
"""
import sys
sys.path.insert("/home/qiule/pies/")

from common.setup import adr
from common.io_utils import load_json, dump_json
from utils import load_models, collect_embs, save_vecs

def main(fin, model_path, temp_dir):
    """"""
    if not temp_dir.exist():
        temp_dir.mkdir(exist_ok=True)
    data = load_json(fin)
    model, tokenizer = load_models(model_path)
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






