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

from imp import reload


from loader import load_data, load_embeddings, assign_emb_dataset
from common.setup import *


def main(fpth, efolder):
    """
    @param fpth: data path
    @param efolder: folder of embeddings
    @return:
    """
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
            regressor = eval(this_reg)
            reg_name = this_reg.split("(")[0]
            each_train(X, Y, loo, regressor, reg_name, epth.stem)


def each_train(X, Y, loo, regressor, reg_no, emb_name):
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


if __name__ == '__main__':
    _path = adr.joinpath("binder")
    fpth = _path.joinpath("Copy of meanRating_July1.xlsx")
    out_dir = _path.joinpath("out4")
    out_dir.mkdir()

    emb_sufix = [".vec", ".word"]
    efolder = tmp_dir  # emb folders
    logging.info("Initialize regressors ... ")
    logging.info("ONLY Ridge & MLP...")
    # regressors = [LinearRegression(), Lasso(alpha=0.1), Ridge(),
    #               RandomForestRegressor(n_estimators=10),
    #               MLPRegressor(hidden_layer_sizes=(50, 10),
    #                            activation='identity', solver='adam', early_stopping=True, max_iter=1000)]
    regressors = ["Ridge()",
                  """MLPRegressor(hidden_layer_sizes=(50, 10), 
                  activation='identity', solver='adam', early_stopping=True, max_iter=1000)"""
                  ]

    main(fpth, efolder)


