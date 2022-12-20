"""

"""
import logging
import json
from dd2 import get_pos_map, pos_tag_canto, pos_tag_mandarin_jiagu, load_sents_parts

logging.basicConfig(level=logging.INFO)
try:
    from google.colab import drive
    drive.mount('/content/drive/')
    # _root = "/content/drive/MyDrive"
    logging.info("Running on Colab ...")
except:
    # _root = "/Users/laniqiu/My Drive"
    logging.info("Running Local ...")

def pos_for_all(out_dir, files, mpth):
    """
    已分词文本做词性标注
    """
    if not out_dir.exists():
        out_dir.mkdir()

    pos_map = get_pos_map(mpth)

    for f in files:
        if f.name.startswith("."):
            continue
        if "simp" in f.name:
            lang, pos_func = "zh", pos_tag_mandarin_jiagu
            upos_map = pos_map
        else:
            lang, pos_func = "zh-hant", pos_tag_canto
            upos_map = None

        fout = out_dir.joinpath(f.name)
        print("lang:", lang)
        sents, all_ = load_sents_parts(f)
        segged = pos_func(sents, upos_map)
        # save to file
        headline = ["sent_id", "word_id", "text", "pos"]
        headline = "\t".join(headline)
        with open(fout, "w", encoding="utf-8") as fw:
            json.dump(segged, fw, ensure_ascii=False)





if __name__ == "__main__":
    from pathlib import Path

    _p = Path("stanza_flow")
    files = _p.glob("*")
    for f in files:
        if f.name.startswith("."):
            continue
















