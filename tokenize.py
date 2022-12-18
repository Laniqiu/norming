import spacy
import jieba
import os
from glob import glob
from collections import Counter
from common.util import general_reader
import numpy as np


def tokenize_and_write(paths, fpth1, fpth2):
    # 下載語言模組
    # spacy.cli.download("zh_core_web_sm")  # 下載 spacy 中文模組
    # spacy.cli.download("en_core_web_sm")  # 下載 spacy 英文模組

    nlp_zh = spacy.load("zh_core_web_sm")  # 載入 spacy 中文模組
    nlp_en = spacy.load("en_core_web_sm")  # 載入 spacy 英文模組

    # 印出前20個停用詞
    # print('--\n')
    # print(f"中文停用詞 Total={len(nlp_zh.Defaults.stop_words)}: {list(nlp_zh.Defaults.stop_words)[:20]} ...")
    # print("--")
    # print(f"英文停用詞 Total={len(nlp_en.Defaults.stop_words)}: {list(nlp_en.Defaults.stop_words)[:20]} ...")

    STOPWORDS = nlp_zh.Defaults.stop_words | \
                nlp_en.Defaults.stop_words | \
                set(["\n", "\r\n", "\t", " ", ""])
    print(len(STOPWORDS))

    def word_segger(
            text: str, token_min_len: int = 1, token_max_len: int = 15, lower: bool = True):
        if lower:
            text = text.lower()
        # text = zhconv.convert(text, "zh-tw")
        return [
            token for token in jieba.cut(text, cut_all=False)
            if token_min_len <= len(token) <= token_max_len and \
               token not in STOPWORDS
        ]

    sents, sents2 = [], []
    for fpth in paths:
        data = general_reader(fpth)
        for line in data:
            segged = word_segger(line.strip().replace("\t", "").replace(" ", ""))
            if segged:
                sents.append(" ".join(segged))
                s = " ".join(list("".join(segged).replace(" ", "")))
                sents2.append(s)
    with open(fpth1, "w") as fw:
        fw.writelines("\n".join(sents))
    with open(fpth2, "w") as fw:
        fw.writelines("\n".join(sents2))


if __name__ == "__main__":

    _dir = "RenMin_Daily"
    paths = glob(os.path.join(_dir, "*.txt"))
    fpth1, fpth2 = "all_segged.txt", "all_tokenized.txt"
    # tokenize_and_write(paths)
    wsents, csents = general_reader(fpth1), general_reader(fpth2)
    words = [w.strip() for line in wsents for w in line.strip().split(" ")]
    chars = [w.strip() for line in csents for w in line.strip().split(" ")]
    # 词频统计
    wcount = sorted(Counter(words).items(), key=lambda x:x[1], reverse=True)
    # 字频统计
    ccount = sorted(Counter(chars).items(), key=lambda x:x[1], reverse=True)
    breakpoint()
    # with open("char_count.txt", "w") as fw:
    #     for (c, count) in ccount:
    #         fw.writelines("{}\t{}\n".format(c, count))
    #
    # with open("word_count.txt", "w") as fw:
    #     for (w, count) in wcount:
    #         fw.writelines("{}\t{}\n".format(w, count))
    # 句长统计（词)
    wscount = np.array([len(s.split(" ")) for s in wsents])
    # mean: 28.2
    # 句长统计（字）
    cscount = np.array([len(s.split(" ")) for s in csents])
    # mean: 63.4



