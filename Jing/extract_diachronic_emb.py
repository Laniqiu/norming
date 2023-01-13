# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 12/1/2023 3:30 pm

"""
from pathlib import Path


from imp import reload
import common
reload(common)
from common.setup import adr, tmp_dir, logging

import utils
reload(utils)
from utils import load_models, collect_embs, save_vecs, get_each_emb
from tqdm import tqdm
from subprocess import check_call
from sklearn.metrics.pairwise import cosine_similarity as cos



def main(fdir, out_dir, model, tokenizer):
    """
    对每一个target的每一个年代的每一个例句取word emb
    @param fdir:
    @param model:
    @param tokenizer:
    @return:
    """
    for folder in tqdm(sorted(fdir.iterdir())):
        if not folder.is_dir():
            continue
        tgt = folder.name  # target word

        # todo 需要删除的代码
        if tgt not in skip_folder:
            print(f"skip folder: {tgt}")
            continue

        def process_one_folder():
            this_outdir = tmp_dir.joinpath(tgt)
            this_outdir.mkdir(parents=True, exist_ok=True)
            for f in tqdm(sorted(folder.glob("*.txt"))):
                fname = f.stem.replace(tgt, "")  # 文件名
                count = 0
                cache = []
                for line in f.open("r"):
                    if tgt not in line:
                        continue
                    dvec = get_each_emb(model, tokenizer, line, tgt)
                    breakpoint()
                #     cache.append(line.strip())
                #     count += 1
                #     if count == chunk_size:
                #         chunk_vecs = collect_embs(tgt, cache, model, tokenizer)
                #         fout = this_outdir.joinpath("{}".format(fname))
                #         logging.info("temporarily saving embeddings at {}".format(fout))
                #         save_vecs(chunk_vecs, fout)
                #         # 更新保存文件名
                #         if len(fname) == 4:
                #             fname += "_0"
                #         else:
                #             fname = fname[:4] + "_{}".format(int(fname[5:])+1)
                #         cache.clear()
                #         count = 0
                # if cache:
                #     chunk_vecs = collect_embs(tgt, cache, model, tokenizer)
                #     fout = this_outdir.joinpath("{}".format(fname))
                #     logging.info("temporarily saving embeddings at {}".format(fout))
                #     save_vecs(chunk_vecs, fout)
                #     cache.clear()


        try:
            process_one_folder()
        except Exception as msg:
            logging.error(msg)


if __name__ == "__main__":

    model_version = "chinese-roberta-wwm-ext"
    model_path = tmp_dir.joinpath(model_version)
    logging.info("loading models from {}".format(model_path))
    model, tokenizer = load_models(model_path)

    # skip_folder = ["下海", "出台", "内伤"]
    skip_folder = ["下海"]
    chunk_size = 150

    svec_dir = tmp_dir.joinpath("svecs")

    main(tmp_dir.joinpath("data_filtered"),  # 输入文件路径
         adr.joinpath("Jing/dvecs"),  # 文件生成保存路径
         model, tokenizer)
