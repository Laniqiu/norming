"""
tokenize, pos tagging, parsing
"""
# colab: import libraries
import stanza
import logging
from collections import OrderedDict

import sys
sys.path.insert(0, "/content/laniqiu/MyDrive/dough")  # colab: import
from dd2 import load_sents_parts

logging.basicConfig(level=logging.INFO)


def stanza_dp(sents, lang, processors='tokenize,pos,lemma,depparse'):
    """
    tokenize -> pos tagging -> parsing
    using stanza
    sents: dict of full sents, sid: full sen
    return: dict of parsed sents, sid: parsed sen
    """
    nlp = stanza.Pipeline(lang=lang, processors=processors)

    res = []
    for sid, sen in sents.items():
        doc = nlp(sen)
        # res[sid] = doc
        for sent in doc.sentences:
            for word in sent.words:
                line = [str(sid), str(word.id), word.text, word.lemma,
                        word.upos, word.xpos, str(word.head), word.deprel]
                line = "\t".join(line) + "\n"
                res.append(line)
        # print(*[
        #     f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}'
        #     for sent in doc.sentences for word in sent.words], sep='\n')
    return res

def main(root, out_dir):
    import os
    from glob import glob
    import json
    import codecs

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    files = glob(os.path.join(root, "*.txt"))
    files.sort()
    for f in files:
        lang = "zh" if "simp" in f else "zh-hant"
        logging.info("language:{}".format(lang))

        sents, all_ = load_sents_parts(f)

        wsents = OrderedDict()  # full sents
        for sid, isen in sents.items():
            wid, sent = zip(*isen)
            wsents[sid] = "".join(sent)
        res = stanza_dp(wsents, lang)

        headline = ["sent_id", "word_id", "text", "lemma", "upos", "xpos", "head", "deprel"]
        res.insert(0, "\t".join(headline))

        fout = f.replace(root, out_dir)
        with codecs.open(fout, "w") as fw:
            fw.writelines(res)


if __name__ == "__main__":
    root = "/Users/laniqiu/My Drive"
    out_dir = "/Users/laniqiu/My Drive/parsed_res"

    main(root, out_dir)

