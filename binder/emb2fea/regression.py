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

import utils
from imp import reload
reload(utils)
from utils import *
from common import logging, get_root


def main(fpth, efolder):
    """
    @param fpth: data path
    @param efolder: folder of embeddings
    @return:
    """
    logging.info("load data from {} ...".format(fpth.name))
    _data = load_data(fpth)  # todo 测试, 后面需要修改

    loo = LeaveOneOut()

    for epth in efolder.glob("*"):
        # if epth.suffix not in emb_sufix:  # in case hidden files exist
        #     continue
        #
        if epth.name not in ["ppmi.wiki.word"]:  # Todo
            continue
        logging.info("load embeddings from {} ...".format(epth.name))
        vectors, dim = load_embeddings(epth, _data)
        X, Y, words = assign_emb_dataset(_data, _data, vectors, dim)
        
        breakpoint()
        logging.info("number of splits {}".format(loo.get_n_splits(X)))
        Spear, Ms, Rm = [], [], []
        for ir, regressor in enumerate(regressors):
            each_train(X, Y, loo, regressor, str(ir), epth.stem)


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
        # for ir, regressor in enumerate(regressors):
        #     logging.info("current regressor no. is {}".format(ir))
        model = regressor.fit(X_train, Y_train)
        Y_pred = model.predict(X_test)
        # mse, rmse = return_MSE_by_Feature(Y_test, Y_pred)  # rmse is the sqrt of mse
        # Ms.append(mse)
        # Rm.append(rmse)
        for i in range(Y_pred.shape[0]):
            Y_output.append(Y_pred[i])
            Y_gold.append(Y_test[i])
    """ evaluation """
    # mse & rmse, with variance
    # Ms_means = np.mean(Ms, axis=0)
    # Rm_means = np.mean(Rm, axis=0)
    # var_mse = np.var(Ms, axis=0)
    # var_rms = np.var(Rm, axis=0)
    # compute spearman across words & features??
    sp_f, sp_w = return_wf_spearman(Y_gold, Y_output)
    # save sp scores
    logging.info("save spearman scores ...")
    fout, wout = out_dir.joinpath("{}_{}_sp_fea.npy".format(emb_name, reg_no)), \
        out_dir.joinpath("{}_{}_sp_word.npy".format(emb_name, reg_no))
    np.save(fout, sp_f)
    np.save(wout, sp_w)


if __name__ == '__main__':
    _root = get_root()
    from pathlib import Path
    _path = Path(_root).joinpath("dough")

    fpth = _path.joinpath("Copy of meanRating_July1.xlsx")
    out_dir = _path.joinpath("sps")
    if not out_dir.exists():
        out_dir.mkdir()
    emb_sufix = [".vec", ".word"]
    efolder = _path.joinpath("norming/embeddings/")  # emb folders
    logging.info("initialize regressors ... ")
    regressors = [LinearRegression(), Lasso(alpha=0.1), Ridge(),
                  RandomForestRegressor(n_estimators=10),
                  MLPRegressor(hidden_layer_sizes=(50, 10),
                               activation='identity', solver='adam', early_stopping=True, max_iter=1000)]

    main(fpth, efolder)

