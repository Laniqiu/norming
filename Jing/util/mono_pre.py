# -*- coding: utf-8 -*-
"""
@author: Lani QIU
@time: 25/2/2023 12:28 pm
从数据中为每个target word随机抽10个例句
"""
import pandas as pd
import codecs
import re
import jieba
import json

from common.setup import adr
from common.io_utils import load_json, dump_json


def func1(files, tgts, fout, record):
    """
    从ccl下载的语料中为每个target抽取10条例句，输出到fout
    例句不足的保存到record文件
    @param files:
    @param tgts:
    @param fout:
    @param record:
    @return:
    """
    out = {}
    ws = []
    for file in files:
        cw = file.stem.split("_")[1:]  # 当前words
        cw = list(set(cw) & set(tgts))  # 去除不在target之内的word
        ws += cw
        out.update(dict(zip(cw, [[]] * len(cw))))
        reg = re.compile("[。！？；](.+?(\[({})\]).+?[。！？；])".format("|".join(cw)))

        # 为每个词找到<=10个例句，写入
        with codecs.open(file, encoding="gbk") as fr:
            for line in fr:
                mm = reg.search(line)
                if not mm:
                    continue
                exp, mw, ww = mm.groups()  # 如果没有匹配到就是哪里出错了。。
                if len(out[ww]) == 10:
                    continue
                # 分词
                seg = jieba.cut(exp.replace("[", "").replace("]", ""), cut_all=False)
                seg_list = [w for w in seg]
                if ww not in seg_list:
                    continue

                out[ww] = out[ww] + [exp]
        sec = []
        for k, v in out.items():
            n = len(v)
            if n < 10:
                line = "{}\t{}\n".format(k, n)
                print("'{}'例句只有{}:".format(k, n))
                sec.append(line)
        if sec:
            with codecs.open(record, "w") as fw:
                fw.writelines(sec)
    # 例句写出
    with codecs.open(fout, encoding="utf-8", mode="w") as fw:
        json.dump(out, fw, ensure_ascii=False, indent=4)
    # 检查是否有遗漏的target
    dif = set(tgts) - set(ws)
    if len(dif) > 0:
        print("这些词需要重新下载例句：")
        print(dif)


def func2(jsons, fout):
    """
    合并多个输出的json文件
    @param jsons:
    @param fout:
    @return:
    """
    dd = {}
    for j in jsons:
        data = load_json(j)
        for k, v in data.items():
            dd[k] = dd.get(k, []) + v

    for k, v in dd.items():
        if len(v) < 10:
            print(k)
            print(len(v))
    dump_json(dd, fout)


def func3(fin, fout):
    """
    检查最终输出的json
    @return:
    """
    data = load_json(fin)
    dd = {}
    for tgt, slist in data.items():
        reg = re.compile("[。！？；](.+?{}.+?[。！？；])".format(tgt))
        slist = list(set(slist))
        if len(slist) < 10:
            print("{}需要再找{}个例句".format(tgt, 10-len(slist)))
        for _, sam in enumerate(slist):
            sam = sam.replace("[{}]".format(tgt), tgt).replace(" ", "")
            mm = reg.search(sam)
            if mm:
                sam = mm.groups()[0]
            slist[_] = sam

        dd[tgt] = slist

    dump_json(dd, fout)




if __name__ == "__main__":
    _dir = adr.joinpath("Jing")
    excel = _dir.joinpath("300targets.xlsx")
    out_dir = _dir.joinpath("100_exps")

    df = pd.read_excel(excel, sheet_name="compare_group", header=None)
    tgts = df.iloc[:, 0]

    # func1(_dir.joinpath("cmp_sam").glob("*.txt"), # 下载的例句txt
    #       tgts, # 目标词汇
    #       out_dir.joinpath("exps_2.json")，  # 例句json输出文件
    #       out_dir.joinpath("sup.txt")  # 记载文件
    #       )

    # func2([out_dir.joinpath("exps_2.json"), out_dir.joinpath("100_all.json")], # 需要整合的文件
    #       out_dir.joinpath("100.json")  # 整合后的输出文件
    #       )

    func3(out_dir.joinpath("100_new.json"),
          out_dir.joinpath("100_.json")
    )

