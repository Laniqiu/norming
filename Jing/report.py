# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 12/1/2023 3:30 pm

"""
from tqdm import tqdm
from threading import Thread
# from multiprocessing import Process
from time import time

import sys
sys.path.insert(0, "/home/qiule/kitchen/pies")

from common.setup import adr, tmp_dir, logging
from common.io_utils import general_writer

from utils import *

# device = "cuda:1"
device = "cpu"


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
    model.to(device)

    # 根据sense vec去找year data， 目标文件夹
    tgt_folder = [f.name.split("_")[0] for f in svec_dir.glob("*") if "_" in f.name]
    tgt_folder = list(set(tgt_folder))

    for tgt in tgt_folder:
        year_folder = fdir.glob("*/{}/*.txt".format(tgt))
        logging.info("Start processing folder: {}".format(tgt))
        ssv = [(spth.stem, load_vecs(spth)) for spth in sorted(svec_dir.glob(f"{tgt}_*"))]
        if not ssv:
            logging.warning("No sense vectors for {}".format(tgt))
            continue
        ss, vv = zip(*ssv)  # sense, sense_vec (i.e., gt)
        mat = [v.cpu().detach().numpy().squeeze() for v in vv]

        this_outdr = out_dir.joinpath(tgt)
        this_outdr.mkdir(parents=True, exist_ok=True)

        # 多进程
        wks = []  # 多线程
        # st = time()
        for year in tqdm(sorted(year_folder)):
            if year.name.startswith("."):  # skip 隐藏文件
                continue
            this_work = Thread(target=process_one_year, args=(year, tgt, this_outdr, model, tokenizer, mat, ss))
            wks.append(this_work)
            this_work.start()
        for wk in wks:
            wk.join()


def process_one_year(*args):
    """

    @param year:
    @param tgt:
    @param args:
    @return:
    """
    year, tgt, this_outdr, model, tokenizer, mat, ss = args

    logging.info("Processing {}".format(year.name))
    fout = this_outdr.joinpath(year.name.replace(tgt, ""))  # 分数输出
    if fout.exists():  # 已经算过 skip
        print("Score already exists, skip {}".format(year.name))
        return
    scores = []
    with year.open("r") as fr:
        lines = fr.readlines()

    for lino, line in enumerate(lines, start=1):
        if tgt not in line:
            logging.warning("No target in {} : {}".format(lino, line))
            scores.append("{}\t{}\n".format(lino, " "))
            continue
        line = line.strip().replace(" ", "")
        try:
            dvec = get_each_emb(model, tokenizer, line, tgt, device=device).reshape(1,
                                                                                    -1).cpu().detach().numpy()
            _s = sim_report(mat, dvec).astype(str).flatten().tolist()
            _s = "\t".join(_s)
        except Exception as msg:
            logging.error("\nTarget:{}\nLino:{}\nErr:{}\n".format(tgt, lino, msg))
            _s = " "
        scores.append("{}\t{}\n".format(lino, _s))  #

    if not scores:
        logging.warning("{} finds no examples in {}".format(tgt, year.stem))
        return
    logging.info("Saving scores at {}...".format(fout.name))
    header = "lino\t{}\n".format("\t".join(ss))
    scores.insert(0, header)
    general_writer(scores, fout)


if __name__ == "__main__":

    # todo 暂时不处理 + 已处理完， 决心好像没整完

    # skip_folder = ["阶级", "批"] + ["建", "节"] + \
    #               ['不断', '中心', '乡', '产业', '介绍', '代表', '企图', '传统', '作', '保障', '党', '内奸', '决定',
    #                '决心', '出口', '出现', '分', '创新', '到底', '办', '努力', '协议', '历史', '原则', '反'] + \
    #               ['公德', '公文', '公证', '反', '合谋', '回', '回复', '姑娘', '富民', '根源', '画卷', '祸害',
    #                '舰只', '荟萃', '较', '过', '金币', '饭厅', "个人"]

    _dir = adr.joinpath("Jing")
    main(adr.joinpath("lfs/chinese_roberta_wwm_ext_pytorch"),  # 模型路径， str
         _dir.joinpath("bert-target-year-data"),  # 年份文件数据
         _dir.joinpath("sens_embs/batch_3"),  # gt数据
         _dir.joinpath("scores/batch_3"),  # 文件生成保存路径
         )

