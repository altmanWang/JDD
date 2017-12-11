import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import SelectPercentile, f_regression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from datetime import datetime
from sklearn.svm import LinearSVR
from sklearn.preprocessing import MaxAbsScaler, Normalizer, StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor

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



# NOTE: Make sure that the class is labeled 'target' in the data file
X = pd.DataFrame()
Y = pd.DataFrame()

for i in range(10, 11):
    x = pd.DataFrame(pd.read_csv("../train/train_x_offline_{}.csv".format(i)))
    y = pd.DataFrame(pd.read_csv("../train/train_y_offline_{}.csv".format(i)))
    X = pd.concat([X, x])
    Y = pd.concat([Y, y])



Test = pd.DataFrame(pd.read_csv("../test/test_x_online.csv"))

X.pop("uid")
Y.pop("uid")
uid = Test.pop("uid")

training_features = X.as_matrix()
training_target = Y.as_matrix()

testing_features = Test.as_matrix()

exported_pipeline = make_pipeline(
    SelectPercentile(score_func=f_regression, percentile=45),
    StackingEstimator(estimator=RidgeCV()),
    MinMaxScaler(),
    GradientBoostingRegressor(loss='ls', alpha=0.9,
                              n_estimators=500,
                              learning_rate=0.02,
                              max_depth=8,
                              subsample=0.8,
                              min_samples_split=20,
                              min_samples_leaf=20,
                              max_leaf_nodes=10)
)

exported_pipeline.fit(training_features, training_target)
y_predict = exported_pipeline.predict(testing_features)


for _ in range(len(y_predict)):
    if y_predict[_] < 0:
        y_predict[_] = 0


result = pd.DataFrame()

result[0] = uid
result[1] = y_predict
result = result.rename(columns={0: "uid", 1: "predict"})

result.to_csv(
    "../result/result_{}_{}_{}_{}_ExtraTreesRegressor.csv".format(datetime.now().month, datetime.now().day, datetime.now().hour,
                                                  datetime.now().minute), header=None, index=False, encoding="utf-8")

