import Load
from dateutil.parser import parse
import pandas as pd
import numpy as np
from math import log
from util import *
def main():
    user = pd.read_csv("../processed_data/processed_user.csv")
    feature = capture_user_information(user)
    uid = user["uid"]
    MONTHs = [11]
    NUMs = [4.0]
    for NUM, MONTH in zip(NUMs, MONTHs):

        #提取
        print("正在提取7/30天各个表格的详细信息")
        loan = pd.read_csv("../processed_data/processed_loan.csv")
        order = pd.read_csv("../processed_data/processed_order.csv")
        for gap in [7,15]:
            order_features = capture_order_information(order, gap, MONTH+1)
            for sub_feature in order_features:
                feature = pd.merge(feature,sub_feature, on = ["uid"], how = "left")
            loan_features = capture_loan_information(loan, gap, MONTH+1)
            for sub_feature in loan_features:
                feature = pd.merge(feature, sub_feature, on=["uid"], how="left")
            order_loan_features = capture_order_loan_cross_information(order, loan, gap, MONTH+1)
            for sub_feature in order_loan_features:
                feature = pd.merge(feature, sub_feature, on=["uid"], how="left")
        # 提取历史贷款信息
        print("正在提取贷款历史统计信息...")
        loan = Load.load_data_csv("../data/processed_loan.csv")
        loan_features = get_loan_feature(loan, MONTH, NUM, uid)
        for sub_feature in loan_features:
            feature = pd.merge(feature, sub_feature, on=["uid"], how="left")
        # 提取购物特征
        print("正在提取购物史统计信息...")
        order = Load.load_data_csv("../data/processed_order.csv")
        order_features = get_order_feature(order, MONTH, NUM, uid)
        for sub_feature in order_features:
            feature = pd.merge(feature, sub_feature, on=["uid"], how="left")
        # 提取点击特征
        print("正在提取点击信息...")
        click = pd.read_csv("../data/click_{}_sum.csv".format(MONTH))
        feature = pd.merge(feature, click, on=["uid"], how="left")


        loan = Load.load_data_csv("../data/processed_loan.csv")
        user = pd.read_csv("../data/t_user.csv")
        new_feature = get_loan_limit_ratio(user ,loan, MONTH)
        feature = pd.merge(feature, new_feature, on=["uid"], how="left")


        #处理异常值
        feature = feature.fillna(0.0)
        # 保存特征数据
        feature.to_csv("../test/test_x_online.csv", index=False)


if __name__ == '__main__':
    main()