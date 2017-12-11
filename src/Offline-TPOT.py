import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler
from tpot.builtins import StackingEstimator, ZeroCount
from datetime import datetime
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.preprocessing import MinMaxScaler
from math import log
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb


def cv(X, Y):
    NUM = 5
    kf = KFold(n_splits=NUM, shuffle=True, random_state=42)
    RMSE = 0

    for train_index, test_index in kf.split(X_features):
        training_features = X_features[train_index, :]
        testing_features = X_features[test_index, :]
        training_target = y_target[train_index]
        testing_target = y_target[test_index]

        # Score on the training set was:-3.237583215412818
        exported_pipeline = make_pipeline(
            SelectPercentile(score_func=f_regression, percentile=45),
            StackingEstimator(estimator=RidgeCV()),
            MinMaxScaler(),
            xgb.XGBRegressor( booster= 'gbtree',objective='reg:linear',max_depth=8, learning_rate=0.02, n_estimators=100,min_child_weight=10),
        )

        exported_pipeline.fit(training_features, training_target)
        y_predict = exported_pipeline.predict(testing_features)
        RMSE += np.round(mean_squared_error(testing_target, y_predict) ** 0.5, 5)
        print(np.round(mean_squared_error(testing_target, y_predict) ** 0.5, 5))
    print("ExtraTreesRegressor: Root mean squared error: %.5f" % (RMSE / NUM))

def offline_test(X, Y):
    training_features, testing_features, training_target, testing_target = \
        train_test_split(X, Y, random_state=42)

    # Score on the training set was:-3.2539972893082396
    exported_pipeline = make_pipeline(
        SelectPercentile(score_func=f_regression, percentile=45),
        StackingEstimator(estimator=RidgeCV()),
        MinMaxScaler(),
        xgb.XGBRegressor(booster='gbtree', objective='reg:linear', max_depth=8, learning_rate=0.02, n_estimators=500,
                         min_child_weight=10)
    )

    exported_pipeline.fit(training_features, training_target)
    results = exported_pipeline.predict(testing_features)
    print("ExtraTreesRegressor: Root mean squared error: %.5f" % np.round(mean_squared_error(testing_target, results) ** 0.5, 5))



# NOTE: Make sure that the class is labeled 'target' in the data file
X = pd.DataFrame()
Y = pd.DataFrame()

for i in range(10, 11):
    x = pd.DataFrame(pd.read_csv("../train/train_x_offline_{}.csv".format(i)))
    y = pd.DataFrame(pd.read_csv("../train/train_y_offline_{}.csv".format(i)))
    X = pd.concat([X, x])
    Y = pd.concat([Y, y])


X.pop("uid")
uid = Y.pop("uid")


X_features = X.as_matrix()
y_target = Y.as_matrix()


offline_test(X_features, y_target)


