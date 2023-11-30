# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:04:00 2023

@author: awei
"""
import argparse

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from __init__ import path
from get_data import data_loading, data_plate 
from feature_engineering import feature_engineering_main

class BuildTrainingProcess:
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2023-02-01', help='进行回测的起始时间')
    parser.add_argument('--date_end', type=str, default='2023-03-01', help='进行回测的结束时间')
    args = parser.parse_args()
    
    print(f'进行回测的起始时间: {args.date_start}\n进行回测的结束时间: {args.date_end}')
    
    # 获取日期段数据
    date_range_data = data_loading.feather_file_merge(args.date_start, args.date_end)
    print(date_range_data)
    
    build_training_process = BuildTrainingProcess
    
    # 特征工程
    perform_feature_engineering = feature_engineering_main.PerformFeatureEngineering()
    date_range_high_bunch, date_range_low_bunch = perform_feature_engineering.feature_engineering_pipline(date_range_data)
    print(date_range_high_bunch)
    
    x, y = date_range_high_bunch.data, date_range_high_bunch.target
    x_df = pd.DataFrame(x, columns= date_range_high_bunch.feature_names)
    x_train, x_test, y_train, y_test = train_test_split(x_df, y, test_size=0.15)
    
    
    # defining parameters 
    params = {
        'task': 'train', # 训练
        'boosting': 'gbdt', # 梯度提升树
        'objective': 'regression', # 回归任务，模型的目标是拟合一个回归函数
        'num_leaves': 10, # 决策树上的叶子节点的数量，控制树的复杂度
        'learning_rate': 0.05, # 学习率，表示每一步迭代中模型权重的调整幅度，影响模型收敛速度
        'metric': ['l2', 'l1'], # 模型评估的指标，这里设置为 L2（均方误差）和 L1（平均绝对误差）
        'verbose': -1, # 控制输出信息的详细程度，-1 表示不输出任何信息
    }

    # laoding data
    lgb_train = lgb.Dataset(x_train, y_train)
    lgb_eval = lgb.Dataset(x_test, y_test, reference=lgb_train)



    # fitting the model
    model = lgb.train(params,
                     train_set=lgb_train,
                     valid_sets=lgb_eval,
                     )#early_stopping_rounds=30

    # prediction
    y_pred = model.predict(x_test)

    # accuracy check
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**(0.5)
    mae = mean_absolute_error(y_test, y_pred)
    print("RMSE: %.2f" % rmse)  # RMSE: 2.77  
    print("MAE: %.2f" % mae)
    
# =============================================================================
# RMSE (Root Mean Squared Error):
#     优点：对较大的误差更加敏感，因为它平方了每个误差值。
#     缺点：对于异常值更为敏感，因为平方可能会放大异常值的影响。
# MAE (Mean Absolute Error):
#     优点：对异常值不敏感，因为它使用的是误差的绝对值。
#     缺点：不像 RMSE 那样对大误差给予更大的权重。
# 选择哪个指标通常取决于你对模型误差的偏好。如果你更关注大误差，可能会选择使用 RMSE。如果你希望对所有误差都保持相对平等的关注，那么 MAE 可能是更好的选择。
# =============================================================================
