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
from common.io_utils import general_reader

import utils
reload(utils)
from utils import load_models, collect_embs, save_vecs,get_each_emb
from tqdm import tqdm
from subprocess import check_call


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
        tgt = folder.name

        # todo
        if tgt in skip_folder:
            print(f"skip folder: {tgt}")
            continue
        # todo

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
                    cache.append(line.strip())
                    count += 1
                    if count == chunk_size:
                        chunk_vecs = collect_embs(tgt, cache, model, tokenizer)
                        fout = this_outdir.joinpath("{}".format(fname))
                        logging.info("temporarily saving embeddings at {}".format(fout))
                        save_vecs(chunk_vecs, fout)
                        # 更新保存文件名
                        if len(fname) == 4:
                            fname += "_0"
                        else:
                            fname = fname[:4] + "_{}".format(int(fname[5:])+1)
                        cache.clear()
                        count = 0
                if cache:
                    chunk_vecs = collect_embs(tgt, cache, model, tokenizer)
                    fout = this_outdir.joinpath("{}".format(fname))
                    logging.info("temporarily saving embeddings at {}".format(fout))
                    save_vecs(chunk_vecs, fout)
                    cache.clear()

            zip_file = "{}.zip".format(out_dir.joinpath(tgt))
            this_cmd = "zip -r {} {}".format(zip_file, this_outdir)
            logging.info(f"zipping {this_outdir} to {zip_file}")
            check_call(this_cmd, shell=True)
            that_cmd = "rm -rf {}".format(this_outdir)
            logging.info(f"deleting {this_outdir}")
            check_call(that_cmd, shell=True)

        try:
            process_one_folder()
        except Exception as msg:
            logging.error(msg)


if __name__ == "__main__":

    model_version = "chinese-roberta-wwm-ext"
    model_path = tmp_dir.joinpath(model_version)
    model, tokenizer = load_models(model_path)

    skip_folder = ["下海", "主旋律", "出台", "内伤"]
    skip_file = [""]
    chunk_size = 150

    main(tmp_dir.joinpath("data_filtered"),  # 输入文件路径
         adr.joinpath("Jing/dvectors"),  # 文件生成保存路径
         model, tokenizer)
