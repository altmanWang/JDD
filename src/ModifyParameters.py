from tpot import TPOTRegressor
from sklearn.model_selection import train_test_split
import pandas as pd


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
    return x_set, y_set, total_prefix




if __name__ == '__main__':
    X = pd.DataFrame()
    Y = pd.DataFrame()
    for i in range(10, 11):
        x = pd.DataFrame(pd.read_csv("../train/train_x_offline_{}.csv".format(i)))
        y = pd.DataFrame(pd.read_csv("../train/train_y_offline_{}.csv".format(i)))
        X = pd.concat([X, x])
        Y = pd.concat([Y, y])



    X.pop("uid")
    Y.pop("uid")
    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        train_size=0.75, test_size=0.25)
    tpot = TPOTRegressor(generations=20, population_size=40, cv=2, verbosity=2)
    tpot.fit(X_train, y_train)
    print(tpot.score(X_test, y_test))
    tpot.export("tpot_boston_pipeline_12.9.py")