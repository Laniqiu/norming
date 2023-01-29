# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 12/1/2023 3:30 pm

"""
from tqdm import tqdm
# from sklearn.metrics.pairwise import cosine_similarity as cos


from imp import reload
import common
reload(common)
from common.setup import adr, tmp_dir, logging
from common.io_utils import general_writer

import utils
reload(utils)
from utils import *


def main(fdir, out_dir, model, tokenizer, dim=768):
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
        if tgt in skip_folder:
            print(f"skip folder: {tgt}")
            continue
        logging.info("Start processing folder: {}".format(tgt))
        # list: (sense, vec)...
        ssv = [(spth.stem, load_vecs(spth)) for spth in sorted(svec_dir.joinpath(tgt).glob("*"))]
        if not ssv:
            logging.warning("No sense vectors for {}".format(tgt))
            continue
        ss, vv = zip(*ssv)
        mat = [v.cpu().detach().numpy().squeeze() for v in vv]
        this_outdr = out_dir.joinpath(tgt)
        this_outdr.mkdir(parents=True, exist_ok=True)
        for year in tqdm(sorted(folder.glob("*.txt"))):
            fout = this_outdr.joinpath(year.name.replace(tgt, ""))  # 分数输出
            if fout.exists():  # todo
                print("skip {}".format(year.name))
                continue
            scores = []
            for lino, line in enumerate(year.open("r")):
                if tgt not in line:
                    logging.warning("No target in {} : {}".format(lino, line))
                    continue
                line = line.strip()
                try:
                    dvec = get_each_emb(model, tokenizer, line, tgt).reshape(1, -1).cpu().detach().numpy()
                    _s = sim_report(mat, dvec).astype(str).flatten().tolist()
                    _s = "\t".join(_s)
                except Exception as msg:
                    logging.error("File:{}\nTarget:{}\nLino:{}\nErr:{}\n".format(year, tgt, lino, msg))
                    _s = " "
                scores.append("{}\t{}\n".format(lino, _s))
            if not scores:
                logging.warning("{} finds no examples in {}".format(tgt, year.stem))
                continue
            logging.info("Saving scores...")
            header = "lino\t{}\n".format("\t".join(ss))
            scores.insert(0, header)
            general_writer(scores, fout)


if __name__ == "__main__":
    model_version = "chinese-roberta-wwm-ext"
    model_path = tmp_dir.joinpath(model_version)
    logging.info("loading models from {}".format(model_path))
    model, tokenizer = load_models(model_path)

    # skip_folder = ["下海", "主旋律"]
    skip_folder = []

    svec_dir = tmp_dir.joinpath("svecs")

    main(tmp_dir.joinpath("data_filtered"),  # 输入文件路径
         adr.joinpath("Jing/scores"),  # 文件生成保存路径
         model, tokenizer)
