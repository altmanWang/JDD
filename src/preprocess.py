__author__ = 'Administrator'
import pandas as pd
from dateutil.parser import parse
from datetime import date
from math import log
import numpy as np
def process_loan(fileName):
    loan = pd.read_csv(fileName)
    loan["loan_amount"] = loan["loan_amount"].apply(lambda x : 5**x - 1)
    loan["month"] = loan["loan_time"].apply(lambda x : parse(x).month)
    loan["loan_time"] = loan["loan_time"].apply(lambda x: str(parse(x).year) + "-" + str(parse(x).month) + "-" + str(parse(x).day))
    loan["is_weekday"] = loan["loan_time"].apply(lambda x : date.isoweekday(parse(x)))
    loan["pay_per_month"] = loan.apply(lambda x : x["loan_amount"] / x["plannum"], axis = 1)
    for i in range(8,13):
        loan["remain_pay_{}".format(i)] = loan.apply(lambda x : x["pay_per_month"] if x["month"]+x["plannum"] >= i and (i - x["month"]) > 0 else 0, axis=1)
        loan["remain_loan_{}".format(i)] = loan.apply(lambda x : x["loan_amount"] - x["pay_per_month"] * (i - x["month"]) if x["month"]+x["plannum"] >= i and (i - x["month"]) > 0else 0.0 ,
                                                      axis=1)
    loan.to_csv("../processed_data/processed_loan.csv", index=False)

def process_user(fileName):
    user = pd.read_csv(fileName)
    user["sex"] = user["sex"].apply(lambda x: 1 if x == 1 else 0)
    limit_one_hot = pd.get_dummies(user["limit"],prefix="limit")
    age_one_hot = pd.get_dummies(user["age"], prefix="age")
    user = user.join(limit_one_hot)
    user = user.join(age_one_hot)
    user.pop("age")
    user.pop("limit")
    user.to_csv("../processed_data/processed_user.csv", index=False)

def process_order(fileName):
    order = pd.read_csv(fileName)
    order["month"] = order["buy_time"].apply(lambda x : parse(x).month)
    order["buy_time"] = order["buy_time"].apply(
        lambda x: str(parse(x).year) + "-" + str(parse(x).month) + "-" + str(parse(x).day))
    order["price"] = order["price"].apply(lambda x: 5 ** x - 1)
    order["discount"] = order["discount"].apply(lambda x: 5 ** x - 1)
    order["price_sum"] = order.apply(lambda x : x["price"]*x["qty"], axis=1)
    order["price_sum_discount"] = order.apply(lambda x : x["price_sum"] - x["discount"] if x["price_sum"] - x["discount"] > 0.0 else 0.0, axis=1)
    order["free"] = order["price"].apply(lambda x : 1 if x ==0 else 0)
    order["discount_ratio"] = order.apply(lambda x :1 -  x["price_sum_discount"] / (x["price_sum"]) if x["price_sum"] != 0 else 0.0, axis=1)
    order.to_csv("../processed_data/processed_order.csv", index=False)

def main():
    process_loan("../data/t_loan.csv")
    #process_user("../data/t_user.csv")
    #process_order("../data/t_order.csv")



if __name__ == '__main__':
    main()