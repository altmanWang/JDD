import Load
from dateutil.parser import parse
import pandas as pd
import numpy as np
from math import log


def split_by_month(data):
    return parse(data).month


def count_price_per_order(column):
    price = column["price"] * column["qty"] - column["discount"]
    if price < 0:
        return 0.0
    return price


def get_pay_per_month(column):
    return column["loan_amount"] / column["plannum"]


def get_remain_loan(column, month):
    tmp = column["loan_amount"] - column["pay_per_month"] * (month - column["month"])
    if tmp >= 0:
        return tmp
    return 0


def get_remain_pay(column, month):
    if month - column["month"] <= column["plannum"] and month - column["month"] > 0:
        return column["pay_per_month"]
    return 0


def get_loan_feature(loan, MONTH):

    current_loan_sum = pd.DataFrame({"loan_sum": loan.loc[loan["month"] == MONTH]["loan_amount"].groupby(
        [loan["uid"]]).sum()}).reset_index()
    current_loan_sum["loan_sum"] = current_loan_sum["loan_sum"].apply(lambda x: log(x + 1, 5))

    features = current_loan_sum
    return features


def main():
    # 提取用户信息
    user = Load.load_data_csv("../data/t_user.csv")
    user.pop("active_date")
    uid = pd.DataFrame(user["uid"])

    MONTH = 11

    # 提取历史贷款信息
    loan = Load.load_data_csv("../data/processed_loan.csv")
    loan_features = get_loan_feature(loan, MONTH)
    feature = pd.merge(uid, loan_features, on=["uid"], how="left")
    # 处理异常值
    feature = feature.fillna(0.0)
    # 保存特征数据
    feature.to_csv("../train/train_y_offline_11.csv", index=False)

if __name__ == '__main__':
    main()