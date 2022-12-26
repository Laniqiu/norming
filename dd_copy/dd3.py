import logging
from dd2 import load_sents_parts
import stanza
from collections import OrderedDict

logging.basicConfig(level=logging.INFO)

def stanza_dp(sents, lang, processors='tokenize,pos,lemma,depparse'):
    """
    tokenize -> pos tagging -> parsing
    using stanza
    """
    nlp = stanza.Pipeline(lang=lang, processors=processors)

    # print for check
    # print(*[
    #     f'id: {word.id}\tword: {word.text}\thead id: {word.head}\thead: {sent.words[word.head - 1].text if word.head > 0 else "root"}\tdeprel: {word.deprel}'
    #     for sent in doc.sentences for word in sent.words], sep='\n')


if __name__ == "__main__":
    import os
    from glob import glob

    root = "/Users/laniqiu/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/" \
           "assignments/dependency_distance/boh/annotator_avg"

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
        stanza_dp(wsents, lang)






