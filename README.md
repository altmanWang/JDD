# JDD-信贷需求预测

本题目希望参赛者通过竞赛数据中的用户基本信息、在移动端的行为数据、购物记录和历史借贷信息来建立预测模型，对未来一个月内用户的借款总金额进行预测。本赛题中包含了各种维度的序列数据、品类交易数据，选手可以采用各种类型的数据预处理算法、模型融合等技术来解决信贷需求这个关键的商业问题。赛题数据为业务情景竞赛数据，所有数据均已进行了采样和脱敏处理，字段取值与分布均与真实业务数据不同。

## 数据集划分
数据集合| 特征提取区间 | 预测区间
---|---|---
线下训练集合| 2016.08.01-2016.10.31 | 2016.11.01-2016.11.30
线上测试集合 | 2016.08.01-2016.11.30 | 2016.12.01-2016.12.31

## 特征工程
给定数据集包括：用户信息表、订单信息表、点击信息表、借款信息表、月结款总信息表。

数据集中关于贷款额、购物单价以及购物折扣做了脱敏处理，导致不能线性相加减，所以需要对其进行处理。脱敏方式：log(x+1,5)

- 用户信息表：
1.  性别
2.	年龄（one-hot编码）
3.	借款初始额度（one-hot编码）
- 订单信息表
1.	个人购物总价平均值与整体购物平均之差
2.	个人购物单价与整体购物单价的平均值之差
3.	每月最低/最高/平均/标准差购买次数
4.	每月最低/最高/最低/标准差总价
5.	每月最低/最高/平均/标准差折扣后总消费
6.	每次消费最低/最高/平均/标准差消费
7.	每次购买的平均折扣
8.	每次购买最低/最高/平均/标准差数量
9.	每次购买的平均折扣力度
10.	每次购买的平均/最低/最高单价
11.	每月免费占比、
12.	每月非免费占比、
13.	每月非免费平均总价格、
14.	每月非免费平均折扣后平均总价格
15.	每月非免费平均折扣力度
16.	每月非免费平均折扣
17.	每次非免费平均单价
18.	当月货物总价格
19.	当月货物平均价格
20.	当月购买物品的平均价格
21.	当月每次平均折扣力度
22.	当月每次平均折扣
23.	总购物次数
24.	免费占比、
25.	非免费占比、
26.	非免费平均总价格、
27.	非免费平均折扣后总价格
28.	非免费平均折扣力度
29.	非免费平均折扣
30.	非免费平均单价
31.	最近7天/最近15天/

1） 非免费平均货物单件、非免费平均折扣率、非免费平均购物量、非免费平均购物总价、非免费平均折扣、非免费平均优惠后购物总价、

2）	平均货物单价、平均折扣率、平均购物量、平均购物总价、平均折扣、平均优惠后购物总价

3）	免费平均购物量

4）	总消费次数、免费商品比例、非免费商品的比例、

5）	购物总价、平均购物总价、总折扣、平均折扣

6）	购物总额占整个月的百分比

- 借款信息表
1.	个人贷款额与总体贷款额平均值之差
2.	个人月供与总体月供平均值之差
3.	贷款概率
4.	连续贷款次数
5.	每一次月供平均/最高/最低/标准差
6.	每一次贷款平均/最高/最低/标准差
7.	每月贷款额平均/最高/最低/标准差
8.	每月月供平均/最高/最低/标准差
9.	历史贷款总额
10.	历史月供总额
11.	还款周期平均/标准差
12.	下月总计月供与下个月总计贷款额
13.	距离下个月贷款时间平均/最高/最低
14.	已经还完多少笔贷款
15.	当月月供与当月贷款总额
16.	当月贷款次数
17.	贷款特征：最近7天/最近15天/最近一个月

1）	平均贷款额、平均月供、贷款次数、平均还款日期、剩余贷款、剩余月供

2）	总贷款额、总月供额

3）	总贷款占当月总贷款比值

- 点击信息表
1.	每月PID点击量
- user表与loan表交叉特征
1.	历史贷款额与初始贷款额比值
- loan表与order表交叉特征
1.	贷款额与购物额比值


## 模型选择与调优
利用TPOP帮助选择模型与参数调优。
通过利用该工具调参，成绩从1.79x变为1.78x.


## 总结
1.	在模型调优与选择中，利用了TPOP获得最终模型（ExtraTreeRegression）。下次应该尝试自己调节参数，并且进行模型融合。（Blagging与Stack）
2.	第一次参加比赛，在特征工程中有很多问题在一开始未发现（例如，无刚量化，数值的统计，对于日期的处理（一开始按月作为时间颗粒度，最后按天作为时间颗粒度），未能发现有效地组合特征（可以利用GBDT树路径来生成新的特征））。
3.	在特征选择上，下次应该尝试做些递归特征删除、利用模型来选择特征（例如LR利用权值来选择）。
4.	线下交叉验证的成绩远远优于先上成绩，可能是特征提取的过程中造成了穿越。
5.	当提取特征遇到瓶颈时，通过调优可以适当地提升分数。
6.	当提取特征造成线下提升，线上下降时，可能该特征与显现测试 相关性太大，导致过拟合线下数据集。
7.	特征工程中注意无量纲化，特征的规格不一样可能导致收敛速度慢（例如购物量与购物单价之间规格不一样）。



###src包含文件主要功能：
- GetTrainData.py:获得线下训练特征
- GetTestData.py:获得线上训练特征
- GenerateLoanSum.py:获得每个月的贷款总额，要用来获得测试集的目标。
- ModifyParametes.py:获得调优模型
- Offline-TPOT.py: 根据ModifyParameters得到的模型来进行线下测试
- Online-TPOT.py; 根据ModifyParameters得到的模型来进行线上测试
- preprocess.py:  对初始表进行处理
- Offline.py: 自己线下调优Offline.py得到的模型进行线上测试