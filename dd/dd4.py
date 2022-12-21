"""
planb:
    加载原始数据(已分词) -> +标注词性 -> 保存
"""

import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)

try:
    from google.colab import drive
    logging.info("Running on Colab ...")
    _root = "/content/drive/MyDrive/"
    sys.path.insert(0, "/content/drive/MyDrive/dough/dd")
    sys.path.insert(0, "/content/drive/MyDrive/dough/")
    # colab 需要安装的packages
    pkgs = ["stanza", "pycantonese", "jiagu", "jieba"]
    for p in pkgs:
        cmd = "pip install {}".format(p)
        os.system(cmd)

except:
    logging.info("Running Local")
    _root = "/Users/laniqiu/My Drive/"

import json
from utils import get_pos_map, load_sents_parts, \
    pos_tag_mandarin_jiagu, pos_tag_canto

def pos_for_all(files, out_dir, mpth):
    """
    对已分词文本做词性标注
    """
    if not out_dir.exists():
        out_dir.mkdir()

    pos_map = get_pos_map(mpth)

    for f in files:
        if "simp" in f.name:
            lang, pos_func = "zh", pos_tag_mandarin_jiagu
            mapping = True
        else:
            lang, pos_func = "zh-hant", pos_tag_canto
            mapping = False

        fout = out_dir.joinpath(f.name)
        print("lang:", lang)
        sents, all_ = load_sents_parts(f)
        segged = pos_func(sents)
        # 简体需要映射pos
        # save to file
        headline = ["sid", "wid", "text", "pos", "upos"]
        headline = "\t".join(headline) + "\n"
        breakpoint()
        with open(fout, "w", encoding="utf-8") as fw:
            json.dump(segged, fw, ensure_ascii=False)


if __name__ == "__main__":
    _p = Path(_root).joinpath("ddata")
    files = _p.joinpath("annotator_avg").glob("*.txt")  # 原始数据
    out_dir = _p.joinpath("posed")  # 词性标注后数据
    mpth = _p.joinpath("upos_map.txt")  # 词性映射字典

    pos_for_all(files, out_dir, mpth)


















