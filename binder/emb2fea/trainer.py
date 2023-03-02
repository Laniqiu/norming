"""
train regression models from Chinese embeddings to the Binder features for the words of your dataset.
steps:
    extract word embeddings
    train regression models
    evaluation
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import LeaveOneOut
import numpy as np
from pathlib import Path

from .loader import load_data, load_embeddings, assign_emb_dataset
from common.setup import logging


regressors = ["Ridge()",
              """MLPRegressor(hidden_layer_sizes=(50, 10), 
              activation='identity', solver='adam', early_stopping=True, max_iter=1000)"""
              ]

def main(fpth, efolder, out_dir, emb_sufix=[".vec", ".word"]):
    """
    @param fpth: data path
    @param efolder: folder of embeddings
    @return:
    """
    fpth = Path(fpth)
    efolder = Path(efolder)
    out_dir = Path(out_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    logging.info("load data from {} ...".format(fpth.name))
    _data = load_data(fpth)

    loo = LeaveOneOut()
    for epth in efolder.glob("*"):
        if epth.suffix not in emb_sufix:  # in case hidden files exist
            continue

        logging.info("load embeddings from {} ...".format(epth.name))
        vectors, dim = load_embeddings(epth, _data)
        X, Y, words = assign_emb_dataset(_data, _data, vectors, dim)
        # save words (en-ch)
        fout = out_dir.joinpath("{}_words.txt".format(epth.stem))
        if not fout.exists():
            words = [f"{e}\t{c}\n" for (e, c) in words]
            logging.info("Saving words at {}".format(fout))
            with open(fout, "w") as fw:
                fw.writelines(words)

        logging.info("number of splits {}".format(loo.get_n_splits(X)))
        for ir, this_reg in enumerate(regressors):
            logging.info("Current regressor:{}".format(this_reg))
            regressor = eval(this_reg)
            reg_name = this_reg.split("(")[0]
            each_train(X, Y, loo, regressor, reg_name, epth.stem, out_dir)


def each_train(X, Y, loo, regressor, reg_no, emb_name, out_dir):
    """
    @param X: data
    @param Y: labels
    @param loo:
    @param regressor:
    @param reg_no: number of regressor, str
    @param emb_name: emb model name, str
    @return:
    """
    Y_output, Y_gold = [], []
    # todo 需要修改，将Y改成每次只有一个，
    """ training with different regressors"""
    for _, (train_index, test_index) in enumerate(loo.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        model = regressor.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)

        Y_output.append(Y_pred)
        Y_gold.append(Y_test)

    # 保存Y_gold, Y_output
    logging.info("saving gt & output...")
    gout = out_dir.joinpath("{}_{}_gold".format(emb_name, reg_no))
    oout = out_dir.joinpath("{}_{}_predict".format(emb_name, reg_no))
    np.save(gout, Y_gold)
    np.save(oout, Y_output)

# if __name__ == '__main__':
#
#     from common.setup import adr
#     print(adr)
#     import sys
#     sys.path.insert(0, "/home/qiule/kitchen/pies")
#
#     main(adr.joinpath("binder/new_ratings.xlsx"),  # 输入文件
#          adr.joinpath("lfs"),  # 模型文件夹
#          adr.joinpath("binder/out"),
#          )
