
import logging
import argparse
import os.path
from datetime import datetime
import logging
import argparse
import os.path
from datetime import datetime
import sys

from gensim.models import word2vec, FastText, KeyedVectors
import multiprocessing

max_cpu_counts = multiprocessing.cpu_count()


# import arguments
parser = argparse.ArgumentParser(description="Build Word Embeddings using Gensim library. It takes a unique corpus \
                                        file in input and build a model using either FastText or Word2Vec library.")
# parser.add_argument("-train", required=True, help="Corpus file (one sentence per line).")
# parser.add_argument("-window", default=4, help="Context window size.\
#                                                 Default FastText/Word2Vec algorithm: 4.")
# parser.add_argument("-size", type=int, default=100, help="Size of word vectors. \
#                                                 Default FastText/Word2Vec algorithm: 100")
# parser.add_argument("-model", type=str, default="cbow", help="Training architecture. Allowed values: `cbow`, `skipgram`.\
#                                                 Default FastText/Word2Vec algorithm: `cbow`.")
# parser.add_argument("-freq", type=int, default=1, help="Ignores all words with total frequency lower than this.\
#                                                 Default FastText/Word2Vec algorithm: 1.")
# parser.add_argument("-out", type=str, help="The directory to save output files. \
#                                                 If not explicit, save outputs in the same folder of the script.")
# group = parser.add_mutually_exclusive_group(required=True)
# group.add_argument("-f", "--fasttext", action="store_true", help="Build FastText model.")
# group.add_argument("-w", "--word2vec", action="store_true", help="Build word2vec model.")

# args = parser.parse_args()


def w2v(iterator, d, w, a, f, out):
    model = word2vec.Word2Vec(iterator, vector_size=d, window=w, min_count=f, workers=max_cpu_counts, sg=a, negative=10)
    # model = word2vec.Word2Vec(iterator, vector_size=d, min_countunt=f, workers=max_cpu_counts, sg=a, negative=10)

    model_vectors = model.wv
    model_vectors.save(out)
    model.save(out)


if __name__ == '__main__':


    # Training algorithm: 1 for skip-gram; otherwise CBOW
    alg = {"cbow": 0, "skipgram": 1}
    fout = "char.w2v"

    fpth = "all_segged.txt"
    fpth2 = "all_tokenized.txt"

    corpus = word2vec.LineSentence(fpth)
    # corpus2 = word2vec.LineSentence(fpth2)
    w2v(corpus, 100, 10, alg['skipgram'], 1, "word.w2v")
    # w2v(corpus2, 100, 10, alg['skipgram'], 1, "char.w2v")
    # wm = "word_200.w2v"
    # cm = "char.w2v"
    # wmodel = KeyedVectors.load(wm)
    # cmodel = KeyedVectors.load(cm)
    breakpoint()

