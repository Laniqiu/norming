# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 12/1/2023 3:30 pm

"""
from tqdm import tqdm

from common.setup import adr, tmp_dir, logging
from common.io_utils import general_writer

from utils import *


def main(model_path, fdir, svec_dir, out_dir, dim=768):
    """
    对每一个target的每一个年代的每一个例句取word emb
    @param fdir:
    @param model:
    @param tokenizer:
    @return:
    """
    logging.info("loading models from {}".format(model_path.name))
    model, tokenizer = load_models(str(model_path))

    for sd_folder in tqdm(sorted(fdir.iterdir())):
        if not sd_folder.is_dir():
            continue
        for folder in tqdm(sorted(sd_folder.iterdir())):
            if not folder.is_dir():
                continue
            tgt = folder.name.strip()  # target word
            # todo 需要删除的代码
            if tgt in skip_folder:
                logging.info(f"Skip folder: {tgt}")
                continue
            logging.info("Start processing folder: {}".format(tgt))
            # list: (sense, vec)...
            # pilot
            # ssv = [(spth.stem, load_vecs(spth)) for spth in sorted(svec_dir.joinpath(tgt).glob("*"))]
            # final
            ssv = [(spth.stem, load_vecs(spth)) for spth in sorted(svec_dir.glob(f"{tgt}_*"))]
            if not ssv:
                logging.warning("No sense vectors for {}".format(tgt))
                continue
            ss, vv = zip(*ssv)  # sense, sense_vec (i.e., gt)
            mat = [v.cpu().detach().numpy().squeeze() for v in vv]
            this_outdr = out_dir.joinpath(tgt)
            this_outdr.mkdir(parents=True, exist_ok=True)
            for year in tqdm(sorted(folder.glob("*.txt"))):
                if year.name.startswith("."):
                    continue
                logging.info("Processing {}".format(year.name))
                fout = this_outdr.joinpath(year.name.replace(tgt, ""))  # 分数输出
                if fout.exists():  # todo
                    print("skip {}".format(year.name))
                    continue
                scores = []
                for i, line in enumerate(year.open("r")):
                    lino = i + 1
                    if tgt not in line:
                        logging.warning("No target in {} : {}".format(lino, line))
                        scores.append("{}\t{}\n".format(lino, " "))
                        continue
                    line = line.strip().replace(" ", "")
                    try:
                        dvec = get_each_emb(model, tokenizer, line, tgt).reshape(1, -1).cpu().detach().numpy()
                        _s = sim_report(mat, dvec).astype(str).flatten().tolist()
                        _s = "\t".join(_s)
                    except Exception as msg:
                        logging.error("Target:{}\nLino:{}\nErr:{}\n".format(tgt, lino, msg))
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

    # todo 暂时不处理 + 已处理完
    skip_folder = ["阶级", "批"] + \
                  ['不断', '中心', '乡', '产业', '介绍', '代表', '企图', '传统', '作', '保障', '党', '内奸', '决定',
                   '决心',
                   '出口', '出现', '分', '创新', '到底', '办', '努力', '协议', '历史', '原则', '反']

    _dir = adr.joinpath("Jing")
    main(adr.joinpath("lfs/chinese_roberta_wwm_ext_pytorch"), # 模型路径
        _dir.joinpath("bert-target-year-data"),  # 年份文件数据
         _dir.joinpath("sens_embs"),  # gt数据
         _dir.joinpath("scores")  # 文件生成保存路径
         )

