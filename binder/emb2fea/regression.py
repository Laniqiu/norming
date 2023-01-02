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

    epths = efolder.glob("*")
    for epth in epths:
        if epth.name.startswith("."):  # in case hidden files exist
            continue
        # todo
        if epth.name not in ["cc.zh.300.vec"]:
            continue
        logging.info("load embeddings from {} ...".format(epth.name))
        vectors, dim = load_embeddings(epth, _data)

        X, Y, words = assign_emb_dataset(_data, _data, vectors, dim)

        loo = LeaveOneOut()
        logging.info("number of splits {}".format(loo.get_n_splits(X)))
        Spear, Ms, Rm = [], [], []
        Y_output, Y_gold = [], []
        """ training with different regressors"""
        for _, (train_index, test_index) in enumerate(loo.split(X)):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]

            model = LinearRegression().fit(X_train, Y_train)
            # model = Ridge().fit(X_train, Y_train)
            # model = RandomForestRegressor(n_estimators=10).fit(X_train, Y_train)
            # model = MLPRegressor(hidden_layer_sizes=(50, 10), activation='identity', solver='adam', early_stopping=True,
            #                      max_iter=1000).fit(X_train, Y_train)
            # model = Lasso(alpha=0.1).fit(X_train, Y_train)

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
        # across features
        Sp_means = return_Spearman_simple(Y_gold, Y_output)
        sp_means = spearman_cof(Y_gold, Y_output)
        breakpoint()


        # print(Ms_means)
        # print(Rm_means)
        # print(Sp_means)


if __name__ == '__main__':
    _root = get_root()

    from pathlib import Path
    _path = Path(_root).joinpath("dough")
    fpth = _path.joinpath("Copy of meanRating_July1.xlsx")
    # epth = _path.joinpath("norming/embeddings/cc.zh.300.vec")
    efolder = _path.joinpath("norming/embeddings/")
    main(fpth, efolder)

