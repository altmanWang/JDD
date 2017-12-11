import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from math import log
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.externals import joblib


def RMSE(estimator, X_test, y_test):
    y_pred = estimator.predict(X_test)
    return mean_squared_error(y_test, y_pred) ** 0.5
#网格搜索GBDT参数
def offline_GBDT_gird(X,Y):
    make_scorer(RMSE, greater_is_better=False)
    x_train = X.as_matrix()
    y_train = Y.as_matrix()
    param = {'n_estimators': range(490, 500, 10)}
    girdSearch = GridSearchCV(estimator=GradientBoostingRegressor(loss='ls', alpha=0.9,
                                             learning_rate=0.02,
                                             max_depth=8,
                                             subsample=0.8,
                                             min_samples_leaf=20,
                                             min_samples_split=20,
                                             max_leaf_nodes=10),
                            param_grid=param, scoring=RMSE, iid=False, cv=5)
    girdSearch.fit(x_train, y_train)
    print(girdSearch.cv_results_)
    print("Best param with grid search: ", girdSearch.best_params_)
    print("Best score with grid search: ", girdSearch.best_score_)
def offline_GBDT_cv(X,Y):
    make_scorer(RMSE, greater_is_better=False)
    x_train = X.as_matrix()
    y_train = Y.as_matrix()
    clf = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                    n_estimators=500,
                                    learning_rate=0.05,
                                    max_depth=8,
                                    subsample=0.8,
                                    min_samples_split=9,
                                    max_leaf_nodes=10)
    this_scores = cross_val_score(clf, x_train, y_train, scoring =RMSE, cv=3, n_jobs = 2)
    return this_scores.mean()
def offline_GBDT_all(X_list, Y_list):
    average_score_test = 0.0
    average_score_train = 0.0
    NUM = 3
    for i in range(NUM):
        print(i)
        predicts_test = []
        evals_test = []
        length_test = 0
        predicts_train = []
        evals_train = []
        length_train = 0
        for x, y in zip(X_list, Y_list):
            x_train = x.as_matrix()
            y_train = y.as_matrix()
            X_dtrain, X_deval, y_dtrain, y_deval = train_test_split(x_train, y_train, random_state=i, test_size=0.2)
            clf = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                            n_estimators=500,
                                            learning_rate=0.05,
                                            max_depth=8,
                                            subsample=0.8,
                                            min_samples_split=9,
                                            max_leaf_nodes=10)

            clf.fit(X_dtrain, y_dtrain)
            predict_test = clf.predict(X_deval)
            predicts_test.append(predict_test)
            evals_test.append(y_deval)
            length_test += len(y_deval)

            predict_train = clf.predict(X_dtrain)
            predicts_train.append(predict_train)
            evals_train.append(y_dtrain)
            length_train += len(y_dtrain)


        y_ture_test = np.ndarray((length_test,))
        y_predict_test = np.ndarray((length_test,))
        tmp = 0
        for true, predict in zip(evals_test, predicts_test):
            y_ture_test[tmp:tmp+len(true)] = true[:,0]
            y_predict_test[tmp:tmp+len(predict)] = predict[:]
            tmp += len(predict)

        for _ in range(len(y_predict_test)):
            if y_predict_test[_] < 0:
                y_predict_test[_] = 0.0
        average_score_test += mean_squared_error(y_ture_test, y_predict_test) ** 0.5

        y_true_train = np.ndarray((length_train,))
        y_predict_train = np.ndarray((length_train,))
        tmp = 0
        for true, predict in zip(evals_train, predicts_train):
            y_true_train[tmp:tmp+len(true)] = true[:,0]
            y_predict_train[tmp:tmp+len(predict)] = predict[:]
            tmp += len(predict)

        for _ in range(len(y_predict_train)):
            if y_predict_train[_] < 0:
                y_predict_train[_] = 0.0
        average_score_train += mean_squared_error(y_true_train, y_predict_train) ** 0.5

    print("GBDT: Train Root mean squared error: %.5f" % np.round(average_score_train / NUM, 5))
    print("GBDT: Test Root mean squared error: %.5f" % np.round(average_score_test/NUM,5) )
def get_GBDT_clf(X_list, Y_list):
    i = 0
    for x, y in zip(X_list, Y_list):
        print(i)
        x_train = x.as_matrix()
        y_train = y.as_matrix()
        X_dtrain, X_deval, y_dtrain, y_deval = train_test_split(x_train, y_train, random_state=2016, test_size=0.2)
        clf = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                        n_estimators=500,
                                        learning_rate=0.02,
                                        max_depth=8,
                                        subsample=0.8,
                                        min_samples_split=10,
                                        max_leaf_nodes=10)

        clf.fit(X_dtrain, y_dtrain)
        joblib.dump(clf, "../model/train_model_{}.m".format(i))
        i += 1
def use_GBDT_clf(X_list, Y_list):
    average_score_test = 0.0
    average_score_train = 0.0
    NUM = 1
    for i in range(NUM):
        print(i)
        predicts_test = []
        evals_test = []
        length_test = 0
        predicts_train = []
        evals_train = []
        length_train = 0
        j = 0
        for x, y in zip(X_list, Y_list):
            x_train = x.as_matrix()
            y_train = y.as_matrix()
            X_dtrain, X_deval, y_dtrain, y_deval = train_test_split(x_train, y_train, random_state=2016, test_size=0.2)
            clf = joblib.load("../model/train_model_{}.m".format(j))
            j += 1
            predict_test = clf.predict(X_deval)
            predicts_test.append(predict_test)
            evals_test.append(y_deval)
            length_test += len(y_deval)

            predict_train = clf.predict(X_dtrain)
            predicts_train.append(predict_train)
            evals_train.append(y_dtrain)
            length_train += len(y_dtrain)


        y_ture_test = np.ndarray((length_test,))
        y_predict_test = np.ndarray((length_test,))
        tmp = 0
        for true, predict in zip(evals_test, predicts_test):
            y_ture_test[tmp:tmp+len(true)] = true[:,0]
            y_predict_test[tmp:tmp+len(predict)] = predict[:]
            tmp += len(predict)

        for _ in range(len(y_predict_test)):
            if y_predict_test[_] < 0:
                y_predict_test[_] = 0.0
        average_score_test += mean_squared_error(y_ture_test, y_predict_test) ** 0.5

        y_true_train = np.ndarray((length_train,))
        y_predict_train = np.ndarray((length_train,))
        tmp = 0
        for true, predict in zip(evals_train, predicts_train):
            y_true_train[tmp:tmp+len(true)] = true[:,0]
            y_predict_train[tmp:tmp+len(predict)] = predict[:]
            tmp += len(predict)

        for _ in range(len(y_predict_train)):
            if y_predict_train[_] < 0:
                y_predict_train[_] = 0.0
        average_score_train += mean_squared_error(y_true_train, y_predict_train) ** 0.5
    print("GBDT: Train Root mean squared error: %.5f" % np.round(average_score_train / NUM, 5))
    print("GBDT: Test Root mean squared error: %.5f" % np.round(average_score_test/NUM,5) )
#LR
def offline_LR(X, Y):
    x_train = X.as_matrix()
    y_train = Y.as_matrix()
    clf = LinearRegression()
    this_scores = cross_val_score(clf, x_train, y_train, scoring =RMSE, cv=5, n_jobs = 2)
    print(this_scores.mean())
    print(this_scores.std())

#XGBT
def offline_XGB_cv(X, Y):
    x_train = X.as_matrix()
    y_train = Y.as_matrix()
    params = {
        'booster': 'dart',
        'objective': 'reg:linear',
        'subsample': 0.8,
        'eta': 0.05,
        'max_depth': 8,
        'seed': 2016,
        'silent': 1,
        'eval_metric': 'rmse',
        'min_child_weight ': 10
    }
    dtrain =  xgb.DMatrix(x_train, y_train)
    result = xgb.cv(params, dtrain=dtrain, num_boost_round=100 ,nfold=5, )
def offline_XGB_grid(X, Y):
    x_train = X.as_matrix()
    y_train = Y.as_matrix()

    X_dtrain, X_deval, y_dtrain, y_deval = train_test_split(x_train, y_train, random_state=1026, test_size=0.3)
    dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
    deval = xgb.DMatrix(X_deval, y_deval)
    watchlist = [(deval, 'eval'), (dtrain, 'training')]
    params = {
        'booster': 'gbtree',
        'objective': 'reg:linear',
        'subsample': 0.8,
        'eta': 0.02,
        'max_depth': 6,
        'seed': 2016,
        'silent': 1,
        'eval_metric': 'rmse',
        'min_child_weight ': 10
    }
    clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
def offline_XGB_all(X_list, Y_list):
    average_score = 0.0
    NUM = 3
    for i in range(NUM):
        predicts = []
        evals = []
        length = 0
        for x, y in zip(X_list, Y_list):
            x_train = x.as_matrix()
            y_train = y.as_matrix()
            x_train, y_train = shuffle(x_train, y_train)
            X_dtrain, X_deval, y_dtrain, y_deval = train_test_split(x_train, y_train, random_state=1026, test_size=0.2)

            dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
            deval = xgb.DMatrix(X_deval, y_deval)
            watchlist = [(deval, 'eval'), (dtrain, 'training')]
            params = {
                'booster': 'gbtree',
                'objective': 'reg:linear',
                'subsample': 0.8,
                'eta': 0.05,
                'max_depth': 8,
                'seed': 2016,
                'silent': 1,
                'eval_metric': 'rmse',
                'min_child_weight ': 10
            }
            clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
            predict = clf.predict(deval)
            predicts.append(predict)
            evals.append(y_deval)
            length += len(y_deval)

        y_ture = np.ndarray((length,))
        y_predict = np.ndarray((length,))
        tmp = 0
        for true, predict in zip(evals, predicts):
            y_ture[tmp:tmp+len(true)] = true[:,0]
            y_predict[tmp:tmp+len(predict)] = predict[:]
            tmp += len(predict)

        for _ in range(len(y_predict)):
            if y_predict[_] < 0:
                y_predict[_] = 0.0
        average_score += mean_squared_error(y_ture, y_predict) ** 0.5
    print("XGB: Root mean squared error: %.5f" % np.round(average_score/NUM,5) )



#RandomForest
def offline_RF_cv(X, Y):
    x_train = X.as_matrix()
    y_train = Y.as_matrix()
    clf = RandomForestRegressor(max_depth=8, min_samples_split=100 , min_samples_leaf=10)
    this_scores = cross_val_score(clf, x_train, y_train, scoring =RMSE, cv=5, n_jobs = 2)
    print(this_scores.mean())
    print(this_scores.std())
#没什么用
def split_by_sex(X, Y, prefixs):
    x_set = []
    y_set = []
    new_prefixs = []

    sex_dic = {"male":0, "female":1}
    for x,y, prefix in zip(X, Y, prefixs):
        for sex in sex_dic:
            split_uid = pd.DataFrame({"uid":x.loc[x["sex"]==sex_dic[sex]]["uid"]})
            split_x = pd.merge(split_uid, x, how="left", on="uid")
            split_y = pd.merge(split_uid, y, how="left", on="uid")
            x_set.append(split_x)
            y_set.append(split_y)
            new_prefixs.append("{}_{}".format(prefix,sex))
    return x_set, y_set, new_prefixs


def split_by_discount(X, Y, prefixs):
    x_set = []
    y_set = []
    new_prefixs = []

    for x,y, prefix in zip(X, Y, prefixs):
        if prefix == "average_loan_negative":
            x_set.append(x)
            y_set.append(y)
            new_prefixs.append(prefix)
        else:
            high_uid = pd.DataFrame({"uid":x.loc[x["average_plannum"]>=1].loc[x["average_plannum"]<3]["uid"]})
            high_x = pd.merge(high_uid, x, on="uid", how="left")
            high_y = pd.merge(high_uid, y, on="uid", how="left")
            x_set.append(high_x)
            y_set.append(high_y)
            new_prefixs.append("{}_{}".format(prefix, "discount_hight"))
            middle_uid = pd.DataFrame({"uid": x.loc[x["average_plannum"] >= 3].loc[x["average_plannum"] < 6]["uid"]})
            middle_x = pd.merge(middle_uid, x, on="uid", how="left")
            middle_y = pd.merge(middle_uid, y, on="uid", how="left")
            x_set.append(middle_x)
            y_set.append(middle_y)
            new_prefixs.append("{}_{}".format(prefix, "discount_middle"))
            low_uid = pd.DataFrame({"uid": x.loc[x["average_plannum"] >= 6]["uid"]})
            low_x = pd.merge(low_uid, x, on="uid", how="left")
            low_y = pd.merge(low_uid, y, on="uid", how="left")
            x_set.append(low_x)
            y_set.append(low_y)
            new_prefixs.append("{}_{}".format(prefix,"discount_low"))
    return x_set, y_set, new_prefixs


def split_by_average_loan(X,Y):
    positive_uid = pd.DataFrame({"uid":X.loc[X["average_loan"] > 0]["uid"]})
    positive_train_x = pd.merge(positive_uid, X, how="left",on="uid")
    positive_train_y = pd.merge(positive_uid, Y, how="left",on="uid")

    negative_uid = pd.DataFrame({"uid": X.loc[X["average_loan"] == 0]["uid"]})
    negative_train_x = pd.merge(negative_uid, X, how="left", on="uid")
    negative_train_y = pd.merge(negative_uid, Y, how="left", on="uid")

    return [positive_train_x, negative_train_x], [positive_train_y, negative_train_y], ["average_loan_positive", "average_loan_negative"]



def split_data(X, Y):
    x_set, y_set, total_prefix = split_by_average_loan(X, Y)
    #x_set, y_set, total_prefix = split_by_discount(x_set, y_set, total_prefix)
    #x_set, y_set, total_prefix = split_by_sex(x_set, y_set, total_prefix)
    return x_set, y_set,total_prefix

def offline_test():
    X = pd.DataFrame()
    Y = pd.DataFrame()
    for i in range(10, 11):
        x = pd.DataFrame(pd.read_csv("../train/train_x_offline_{}.csv".format(i)))
        y = pd.DataFrame(pd.read_csv("../train/train_y_offline_{}.csv".format(i)))
        X = pd.concat([X, x])
        Y = pd.concat([Y, y])

    x_set, y_set, names = split_data(X, Y)
    for x, y in zip(x_set, y_set):
        x.pop("uid")
        y.pop("uid")

    offline_GBDT_all(x_set, y_set)

    X.pop("uid")
    Y.pop("uid")

    x_set.append(X)
    y_set.append(Y)
    names.append("ALL")

    for x, y, name in zip(x_set, y_set, names):
        mean = offline_GBDT_cv(x, y)
        print("{} RMSE mean score of CV : {}".format(name, np.round(mean, 5)))



def main():
    offline_test()
    '''
    X = pd.DataFrame()
    Y = pd.DataFrame()
    for i in range(10, 11):
        x = pd.DataFrame(pd.read_csv("../train/train_x_offline_{}.csv".format(i)))
        y = pd.DataFrame(pd.read_csv("../train/train_y_offline_{}.csv".format(i)))
        X = pd.concat([X, x])
        Y = pd.concat([Y, y])
    X.pop("uid")
    Y.pop("uid")
    offline_XGB_grid(X, Y)
    '''
if __name__ == '__main__':
    main()