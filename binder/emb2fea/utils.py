"""
utils
"""
import pandas as pd
import numpy as np

try:
    from google.colab import drive
    # drive.mount('/content/drive')
    _root = "/content/drive/MyDrive/"
except:
    _root = "/Users/laniqiu/My Drive/"


def load_data(fname, chinese=True):
    """

    @param fname:
    @param chinese: 是否中文数据集
    @return:
    """
    ds = {}
    matrix = []

    df = pd.read_excel(fname)

    # encoding is utf-8 when dealing with Chinese data
    for i in range(5, len(df.words)):
        word = df.words[i]
        ds[word] = i - 5

        vector = df.iloc[i].iloc[11:79].values
        matrix.append(np.array(vector, dtype=float))

    ds['_matrix_'] = np.around((np.array(matrix)), decimals=4)

    return ds


def load_embeddings(fname, ds_words):
    """
    @param fname: path of word embeddings
    @param ds_words:
    @return:
    """
    emb = {}
    matrix = []
    dims = 0
    with open(fname, 'r', encoding='utf-8', errors="ignore") as f:
        for line in f:
            line = line.strip().split()
            if dims == 0:
                if len(line) == 2:
                    continue
                else:
                    dims = len(line) - 1

            word = line[0]

            if word not in ds_words:
                continue

            if word in ['', ' ', '\t', '\n']:
                print('Word {} has no value.'.format(word))
                continue
            try:
                vec = [float(x) for x in line[1:]]
                if len(vec) == dims:
                    arrays = np.array(vec)
                else:
                    continue
            except:
                continue
            if word in emb.keys():
                continue

            emb[word] = len(matrix)
            matrix.append(arrays)

    print(len(matrix))
    emb['_matrix_'] = np.array(matrix)  # normalize()

    return emb, dims


if __name__ == '__main__':
    from pathlib import Path
    _path = Path(_root)