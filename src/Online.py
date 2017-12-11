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

def offline_GBDT_all(X_list, Y_list, test_set, uid):
    predicts = []
    length = 0
    for x, y, test in zip(X_list, Y_list, test_set):
        x.pop("uid")
        y.pop("uid")
        test.pop("uid")

        x_train = x.as_matrix()
        y_train = y.as_matrix()
        clf = GradientBoostingRegressor(loss='ls', alpha=0.9,
                                        n_estimators=500,
                                        learning_rate=0.05,
                                        max_depth=8,
                                        subsample=0.8,
                                        min_samples_split=9,
                                        max_leaf_nodes=10)
        clf.fit(x_train, y_train)
        predict = clf.predict(test)
        predicts.append(predict)
        length += len(test)

    y_predict = np.ndarray((length,))
    tmp = 0
    for predict in predicts:
        y_predict[tmp:tmp+len(predict)] = predict[:]
        tmp += len(predict)
    for _ in range(len(y_predict)):
        if y_predict[_] < 0:
            y_predict[_] = 0.0

    result = pd.DataFrame()

    result[0] = uid
    result[1] = y_predict
    result = result.rename(columns={0:"uid",1:"predict"})

    result.to_csv("../result/result_{}_{}_{}_{}GBDT.csv".format(datetime.now().month,datetime.now().day,datetime.now().hour,datetime.now().minute ), header=None, index=False, encoding="utf-8")
def use_GBDT_model(test_set, uid):
    predicts = []
    length = 0
    j = 0
    for test in test_set:
        test.pop("uid")
        clf = joblib.load("../model/train_model_{}.m".format(j))
        predict = clf.predict(test)
        predicts.append(predict)
        length += len(test)
        j += 1
    y_predict = np.ndarray((length,))
    tmp = 0
    for predict in predicts:
        y_predict[tmp:tmp+len(predict)] = predict[:]
        tmp += len(predict)
    for _ in range(len(y_predict)):
        if y_predict[_] < 0:
            y_predict[_] = 0.0
    result = pd.DataFrame()
    result[0] = uid
    result[1] = y_predict
    result.to_csv("../result/result_{}-{}_{}_{}_GBDT.csv".format(datetime.now().month,datetime.now().day,datetime.now().hour,datetime.now().minute ), header=None, index=False, encoding="utf-8")
def split_by_average_loan(X,Y, Test):
    positive_uid = pd.DataFrame({"uid":X.loc[X["average_loan"] > 0]["uid"]})
    positive_train_x = pd.merge(positive_uid, X, how="left",on="uid")
    positive_train_y = pd.merge(positive_uid, Y, how="left",on="uid")

    negative_uid = pd.DataFrame({"uid": X.loc[X["average_loan"] == 0]["uid"]})
    negative_train_x = pd.merge(negative_uid, X, how="left", on="uid")
    negative_train_y = pd.merge(negative_uid, Y, how="left", on="uid")

    positive_test = pd.DataFrame(Test.loc[Test["average_loan"] > 0])
    negative_test = pd.DataFrame(Test.loc[Test["average_loan"] == 0])


    return [positive_train_x, negative_train_x], [positive_train_y, negative_train_y], [positive_test, negative_test]
def split_data(X, Y, Test):
    x_set, y_set, test = split_by_average_loan(X, Y, Test)
    return x_set, y_set,test

def main():
    X = pd.DataFrame()
    Y = pd.DataFrame()

    for i in range(10, 11):
        x = pd.DataFrame(pd.read_csv("../train/train_x_offline_{}.csv".format(i)))
        y = pd.DataFrame(pd.read_csv("../train/train_y_offline_{}.csv".format(i)))
        X = pd.concat([X, x])
        Y = pd.concat([Y, y])

    Test = pd.DataFrame(pd.read_csv("../test/test_x_online.csv"))
    #x_set, y_set, test_set = split_data(X, Y, Test)

    x_set = [X]
    y_set = [Y]
    test_set = [Test]

    Test_new = pd.DataFrame()
    for test in test_set:
        Test_new = pd.concat([Test_new, test])
    uid = Test_new.pop("uid")

    offline_GBDT_all(x_set, y_set, test_set, uid)
if __name__ == '__main__':
    main()