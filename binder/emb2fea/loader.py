"""
utils
"""
import numpy as np
import pandas as pd


def load_data(fname, irow=5, icol=11, ecol=79):
    """

    @param fname: 数据集文件名，xlsx
    @param irow: 从irow行开始
    @param icol: 从icol列开始
    @param ecol: 从ecol列结束
    @param chinese: 是否为中文数据集
    @return:
    """
    des = {}
    matrix = []

    df = pd.read_excel(fname)

    # encoding is utf-8 when dealing with Chinese data
    for i in range(irow, len(df.words)):
        word = df.words[i]
        eword = df.EngWords[i]
        des[(eword, word)] = i - irow

        vector = df.iloc[i].iloc[icol:ecol].values
        matrix.append(np.array(vector, dtype=float))

    des['_matrix_'] = np.around((np.array(matrix)), decimals=4)

    return des

def generate_random_embs(ds_words, rand_dims=300):
    emb = {}
    matrix = []
    dims = 0
    keys = [key for key in ds_words if key not in ['domains', '_matrix_']]
    _, tgts = zip(*keys)
    # generate random embs
    for word in tgts:
        arrays = np.random.random(rand_dims)
        emb[word] = len(matrix)
        matrix.append(arrays)
    emb['_matrix_'] = np.array(matrix)
    return emb, rand_dims

def load_embeddings(fname, ds_words, rand_dims=0):
    """
    @param fname: path of word embeddings
    @param ds_words: target words
    @return:
    """
    emb = {}
    matrix = []
    dims = 0
    keys = [key for key in ds_words if key not in ['domains', '_matrix_']]
    _, tgts = zip(*keys)


    with open(fname, 'r', encoding='utf-8', errors="ignore") as f:
        for line in f:
            line = line.strip().split()
            if dims == 0:
                if len(line) == 2:
                    continue
                else:
                    dims = len(line) - 1

            word = line[0]
            # if word not in ds_words:
            if word not in tgts:
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

            emb[word] = len(matrix)  # as indices
            matrix.append(arrays)

    print(len(matrix))
    emb['_matrix_'] = np.array(matrix)  # normalize()

    return emb, dims


def get_vec(word, embs, PRINT_WORD=False):
    """

    @param word:
    @param embs:
    @param PRINT_WORD:
    @return:
    """
    if word in embs.keys():
        try:
            return embs['_matrix_'][embs[word], :]
        except:
            if word != "_matrix_":
                print('{} should have been there but something went wrong when loading it!'.format(word))
            return []
    else:
        if PRINT_WORD:
            print('{} not in the dataset.'.format(word))

        # return np.zeros(embs['_matrix_'][0].shape[0])
        return []


def assign_emb_dataset(ds, ds_words, embs, dim, norm_dim=68):
    """
    @param ds: dataset
    @param ds_words: words
    @param embs:
    @param dim:
    @param norm_dim: size of attributes/ features
    @return:
        X: dataset for training and validation
        Y: labels
        words: words
    """

    words, X, Y = [], [], []
    for i, key in enumerate(ds_words):
        if key in ['domains', '_matrix_']:
            continue
        eword, word = key  # 英文词、中文词

        if word not in embs:
            print('Word {} does not appear in embs'.format(word))
            continue

        words.append(key)

        vec = get_vec(word, embs)
        norm = get_vec(key, ds)

        if len(vec) != dim or len(norm) != norm_dim:
            continue

        X.append(vec)
        Y.append(norm)
    X, Y = np.array(X), np.array(Y)
    return X, Y, words


if __name__ == "__main__":
    from pathlib import Path
    fpth = "/Users/laniqiu/My Drive/dough/sps"
    fout = "/Users/laniqiu/Library/CloudStorage/OneDrive-TheHongKongPolytechnicUniversity/assignments/binder/" \
           "spearman_lasso_only.txt"
    files = sorted(Path(fpth).glob("*_fea.npy"))






