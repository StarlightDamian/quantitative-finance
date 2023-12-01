# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:04:00 2023

@author: awei
"""
import argparse

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from __init__ import path
from get_data import data_loading #, data_plate 
from feature_engineering import feature_engineering_main

class StockPredictionModel:
    def __init__(self):
        self.perform_feature_engineering = feature_engineering_main.PerformFeatureEngineering()
        self.model = None

    def feature_engineering_pipline(self, date_range_data):
        date_range_high_bunch, date_range_low_bunch = self.perform_feature_engineering.feature_engineering_dataset_pipline(date_range_data)
        x, y = date_range_high_bunch.data, date_range_high_bunch.target
        x_df = pd.DataFrame(x, columns=date_range_high_bunch.feature_names)
        x_train, x_test, y_train, y_test = train_test_split(x_df, y, test_size=0.15)
        return x_train, x_test, y_train, y_test

    def train_model(self, x_train, y_train):
        # defining parameters
        params = {
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression',
            'num_leaves': 10,
            'learning_rate': 0.05,
            'metric': ['mae', 'root_mean_squared_error'],
            'verbose': -1,
        }

        # loading data
        lgb_train = lgb.Dataset(x_train, y_train)

        # fitting the model
        self.model = lgb.train(params, train_set=lgb_train)

    def evaluate_model(self, x_test, y_test):
        # prediction
        y_pred = self.model.predict(x_test)

        # accuracy check
        mse = mean_squared_error(y_test, y_pred)
        rmse = mse ** (0.5)
        mae = mean_absolute_error(y_test, y_pred)
        print("RMSE: %.2f" % rmse)
        print("MAE: %.2f" % mae)
        
        result_y = pd.DataFrame([y_test,y_pred]).T.rename(columns={0: '明天_最高价_百分比_真实值', 1: '明天_最高价_百分比_预测值'})
        result_x = x_test[['high', 'low', 'close']].rename(columns={'high': '最高价', 'low': '最低价', 'close': '今收盘价'})
        #result_x.loc[:, '备注'] = '封板'
        result_x['备注'] = result_x.apply(lambda row: '封板' if row['最高价'] == row['最低价'] else '', axis=1)#, inplace=True
        
        result_x = result_x.reset_index(drop=True)
        result_check = pd.concat([result_y, result_x], axis=1)
        print(result_check)
        result_check.to_csv(f'{path}/data/result_check.csv')
        
        return rmse, mae

    def save_model(self, model_path):
        # Save the model to a file
        self.model.save_model(model_path)

    def load_model(self, model_path):
        # Load the model from a file
        self.model = lgb.Booster(model_file=model_path)

    def plot_feature_importance(self):
        # Plot feature importance
        lgb.plot_importance(self.model, importance_type='split', figsize=(10, 6), title='Feature importance (split)')
        lgb.plot_importance(self.model, importance_type='gain', figsize=(10, 6), title='Feature importance (gain)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2022-03-01', help='进行回测的起始时间')
    parser.add_argument('--date_end', type=str, default='2023-03-01', help='进行回测的结束时间')
    args = parser.parse_args()

    print(f'进行回测的起始时间: {args.date_start}\n进行回测的结束时间: {args.date_end}')

    # 获取日期段数据
    date_range_data = data_loading.feather_file_merge(args.date_start, args.date_end)

    stock_model = StockPredictionModel()

    x_train, x_test, y_train, y_test = stock_model.feature_engineering_pipline(date_range_data)
    stock_model.train_model(x_train, y_train)
    stock_model.evaluate_model(x_test, y_test)

    # Save the model
    model_path = f'{path}/checkpoint/stock_prediction_model.txt'
    stock_model.save_model(model_path)

    # Load the model
    stock_model.load_model(model_path)

    # Plot feature importance
    stock_model.plot_feature_importance()


#w×RMSE+(1−w)×MAE
# =============================================================================
# import argparse
# 
# import lightgbm as lgb
# import pandas as pd
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_absolute_error
# 
# from __init__ import path
# from get_data import data_loading, data_plate 
# from feature_engineering import feature_engineering_main
# 
# class BuildTrainingProcess:
#     def __init__(self):
#         self.perform_feature_engineering = feature_engineering_main.PerformFeatureEngineering()
#         
#     def train_pipline(self, date_range_data):
#         date_range_high_bunch, date_range_low_bunch = self.perform_feature_engineering.feature_engineering_pipline(date_range_data) # 特征工程
#         
#         
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     # parser.add_argument('--date_start', type=str, default='2017-02-01', help='进行回测的起始时间')
#     parser.add_argument('--date_start', type=str, default='2022-03-01', help='进行回测的起始时间')
#     parser.add_argument('--date_end', type=str, default='2023-03-01', help='进行回测的结束时间')
#     args = parser.parse_args()
#     
#     print(f'进行回测的起始时间: {args.date_start}\n进行回测的结束时间: {args.date_end}')
#     
#     # 获取日期段数据
#     date_range_data = data_loading.feather_file_merge(args.date_start, args.date_end)
#     # print(date_range_data)
#     
#     #build_training_process = BuildTrainingProcess()
#     #build_training_process.train_pipline(date_range_data)
# 
#     
#     # print(date_range_high_bunch)
#     perform_feature_engineering = feature_engineering_main.PerformFeatureEngineering()
#     date_range_high_bunch, date_range_low_bunch = perform_feature_engineering.feature_engineering_pipline(date_range_data) # 特征工程
#     x, y = date_range_high_bunch.data, date_range_high_bunch.target
#     x_df = pd.DataFrame(x, columns= date_range_high_bunch.feature_names)
#     x_train, x_test, y_train, y_test = train_test_split(x_df, y, test_size=0.15)
#     
#     
#     # defining parameters
#     params = {
#         'task': 'train', # 训练
#         'boosting': 'gbdt', # 梯度提升树
#         'objective': 'regression', # 回归任务，模型的目标是拟合一个回归函数
#         'num_leaves': 10, # 决策树上的叶子节点的数量，控制树的复杂度
#         'learning_rate': 0.05, # 学习率，表示每一步迭代中模型权重的调整幅度，影响模型收敛速度
#         'metric': ['l2', 'l1'], # 模型评估的指标，这里设置为 L2（均方误差）和 L1（平均绝对误差）
#         'verbose': -1, # 控制输出信息的详细程度，-1 表示不输出任何信息
#     }
# 
#     # laoding data
#     lgb_train = lgb.Dataset(x_train, y_train)
#     lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)
# 
# 
# 
#     # fitting the model
#     model = lgb.train(params,
#                      train_set=lgb_train,
#                      valid_sets=lgb_eval,
#                      )#early_stopping_rounds=30
# 
#     # prediction
#     y_pred = model.predict(x_test)
# 
#     # accuracy check
#     mse = mean_squared_error(y_test, y_pred)
#     rmse = mse**(0.5)
#     mae = mean_absolute_error(y_test, y_pred)
#     print("RMSE: %.2f" % rmse)  
#     print("MAE: %.2f" % mae)
#     
#     
#     result_y = pd.DataFrame([y_test,y_pred]).T.rename(columns={0: '明天_最高价_真实值', 1: '明天_最高价_预测值'})
#     result_x = x_test[['high', 'low', 'close']].rename(columns={'high': '最高价', 'low': '最低价', 'close': '今收盘价'})
#     result_x = result_x.reset_index(drop=True)
#     result_check = pd.concat([result_y, result_x], axis=1)
#     print(result_check)
#     result_check.to_csv(f'{path}/data/result_check.csv')
# =============================================================================
    
    
# =============================================================================
#     Index(['dateDiff', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn',
#            'tradestatus', 'pctChg', 'isST', 'industry_交通运输', 'industry_休闲服务',
#            'industry_传媒', 'industry_公用事业', 'industry_其他', 'industry_农林牧渔',
#            'industry_化工', 'industry_医药生物', 'industry_商业贸易', 'industry_国防军工',
#            'industry_家用电器', 'industry_建筑材料', 'industry_建筑装饰', 'industry_房地产',
#            'industry_有色金属', 'industry_机械设备', 'industry_汽车', 'industry_电子',
#            'industry_电气设备', 'industry_纺织服装', 'industry_综合', 'industry_计算机',
#            'industry_轻工制造', 'industry_通信', 'industry_采掘', 'industry_钢铁',
#            'industry_银行', 'industry_非银金融', 'industry_食品饮料'],
#           dtype='object')
# =============================================================================
    
# =============================================================================
# RMSE (Root Mean Squared Error):
#     优点：对较大的误差更加敏感，因为它平方了每个误差值。
#     缺点：对于异常值更为敏感，因为平方可能会放大异常值的影响。
# MAE (Mean Absolute Error):
#     优点：对异常值不敏感，因为它使用的是误差的绝对值。
#     缺点：不像 RMSE 那样对大误差给予更大的权重。
# 选择哪个指标通常取决于你对模型误差的偏好。如果你更关注大误差，可能会选择使用 RMSE。如果你希望对所有误差都保持相对平等的关注，那么 MAE 可能是更好的选择。
# =============================================================================
