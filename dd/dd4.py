"""
planb:
    加载原始数据(已分词) -> +标注词性 -> 保存
"""

import logging
logging.basicConfig(level=logging.INFO)

try:
    from google.colab import drive
    logging.info("Running on Colab ...")
    _root = "/content/drive/MyDrive/"
except:
    logging.info("Running Local")
    _root = "/Users/laniqiu/My Drive/"


from otr_utils import get_pos_map, load_sents_parts, \
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
        else:
            lang, pos_func = "zh-hant", pos_tag_canto
        print("lang:", lang)
        fout = out_dir.joinpath(f.name)
        sents, all_ = load_sents_parts(f)
        segged = pos_func(sents)
        # 简体需要映射pos
        # save to file
        headlist = ["sid", "wid", "text", "pos", "upos"]
        headline = "\t".join(headlist) + "\n"
        outt = [headline]
        for sid, values in segged.items():
            for wid, word, pos in values:
                if lang == "zh-hant":
                    upos = pos
                else:
                    upos = pos_map[pos]
                line = "{}\t{}\t{}\t{}\t{}\n".format(sid, wid, word, pos, upos)
                outt.append(line)
        with open(fout, "w", encoding="utf-8") as fw:
            fw.writelines(outt)

def add_jpos(files):
    """add jieba pos to original files"""

    import jieba.posseg as tagger  #

    for f in files:
        print(f.name)
        data = f.open().readlines()
        for _, line in enumerate(data):
            if _ == 0:
                data[_] = line.strip() + "\t{}\n".format("jpos")
                continue
            sid, wid, text, pos, upos = line.strip().split("\t")
            jposs = [p for (w, p) in tagger.cut(text)]
            jpos = jposs[0] if len(jposs) == 1 else "-"
            data[_] = line.strip() + "\t{}\n".format(jpos)
        with open(f, "w") as fw:
            fw.writelines(data)


def check_(files, jmpth):
    """
    check and correct some pos tags
    """
    pass


def add_xpos(xfiles, pdir):
    """
    add xpos to posed files
    xfiles: stanza processed files
    pdir: dir of posed files"""

    import pandas as pd
    for xf in xfiles:
        data = pd.read_table(xf, sep="\t").values[:, 3:6]
        udict = dict(zip(data[:, 0], data[:, 1]))
        xdict = dict(zip(data[:, 0], data[:, 2]))
        assert(len(udict) == len(xdict))
        fout = pdir.joinpath(xf.name)
        print(fout)
        data = fout.open("r").readlines()
        for idx, line in enumerate(data):
            if idx == 0:
                data[idx] = line.strip() + "\t{}\n".format("xpos")
                continue
            sid, wid, text, pos, upos, jpos, apos = line.strip().split("\t")
            if text in udict and udict[text] == apos:
                xpos = xdict[text]
            else:
                xpos = ""
            data[idx] = "\t".join([sid, wid, text, pos, upos, jpos, apos, xpos]) + "\n"
        with open(fout, "w") as fw:
            fw.writelines(data)




if __name__ == "__main__":
    from pathlib import Path
    _p = Path(_root).joinpath("ddata")
    files = _p.joinpath("annotator_avg").glob("*.txt")  # 原始数据
    out_dir = _p.joinpath("posed")  # 词性标注后数据
    mpth = _p.joinpath("upos_map.txt")  # 词性映射字典

    pos_for_all(files, out_dir, mpth)


















