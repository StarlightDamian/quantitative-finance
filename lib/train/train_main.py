# -*- coding: utf-8 -*-
"""
Created on Thu Aug 24 16:04:00 2023

@author: awei
"""
from datetime import datetime
import argparse
from sqlalchemy import create_engine

import joblib
import numpy as np
from sklearn.multioutput import MultiOutputRegressor
import seaborn as sns
import matplotlib.pyplot as plt
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

from __init__ import path
#from get_data import data_loading #, data_plate
from feature_engineering import feature_engineering_main
from base import base_connect_database

# Constants
MULTIOUTPUT_MODEL_PATH = f'{path}/checkpoint/prediction_stock_lgbm_model.joblib'
PREDICTION_PRICE_OUTPUT_CSV_PATH = f'{path}/data/prediction_stock_price_train.csv'
TRAIN_TABLE_NAME = 'prediction_stock_price_train'
EVAL_TABLE_NAME = 'eval_train'
TEST_SIZE = 0.15 #数据集分割中测试集占比
#SEED = 2023 #保证最大值和最小值的数据部分保持一致

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class StockTrainModel:
    def __init__(self):
        """
        Initialize StockTrainModel object, including feature engineering and database connection.
        """
        self.perform_feature_engineering = feature_engineering_main.PerformFeatureEngineering()
        self.model_multioutput_regressor = None
        
        params = {
            'task': 'train',
            'boosting': 'gbdt',
            'objective': 'regression',
            'num_leaves': 10, # 决策树上的叶子节点的数量，控制树的复杂度
            'learning_rate': 0.05,
            'metric': ['mae'], # 模型通过mae进行优化, root_mean_squared_error进行评估。, 'root_mean_squared_error'
            'verbose': -1, # 控制输出信息的详细程度，-1 表示不输出任何信息
        }
        # loading data
        lgb_regressor = lgb.LGBMRegressor(**params)
        self.model_multioutput_regressor = MultiOutputRegressor(lgb_regressor)
        
    def load_dataset(self, date_range_bunch, test_size=TEST_SIZE):#, random_state=SEED
        x, y = date_range_bunch.data, date_range_bunch.target
        x_df = pd.DataFrame(x, columns=date_range_bunch.feature_names)
        x_train, x_test, y_train, y_test = train_test_split(x_df, y, test_size=test_size)#, random_state=random_state
        x_test = x_test.reset_index(drop=True)  # The index is retained during the train_test_split process. The index is reset in this step.
        return x_train, x_test, y_train, y_test
    
    def feature_engineering_train_pipline(self, date_range_data):
        """
        Execute feature engineering pipeline and return datasets for training and testing.
        """
        date_range_bunch = self.perform_feature_engineering.feature_engineering_dataset_pipline(date_range_data)
        #print('date_range_bunch', date_range_bunch)
        x_train, x_test, y_train, y_test = self.load_dataset(date_range_bunch)
        return x_train, x_test, y_train, y_test

    def field_handle(self, y_result, x_test):

        y_result = y_result.rename(columns={0: 'rear_low_real',
                                        1: 'rear_high_real',
                                        2: 'rear_diff_real',
                                        3: 'rear_low_pred',
                                        4: 'rear_high_pred',
                                        5: 'rear_diff_pred',})
        
        prediction_stock_price  = pd.concat([y_result, x_test], axis=1)
        
        prediction_stock_price['remarks'] = prediction_stock_price.apply(lambda row: 'limit_up' if row['high'] == row['low'] else '', axis=1)
        
        prediction_stock_price = prediction_stock_price[['rear_low_real', 'rear_low_pred', 'rear_high_real', 'rear_high_pred', 'rear_diff_real', 'rear_diff_pred',
                                                         'open','high', 'low', 'close','volume','amount','turn', 'pctChg', 'remarks']]
                                                         


        return prediction_stock_price

    def data_processing_pipline(self, date_range_data):
        x_train, x_test, y_train, y_test = self.feature_engineering_train_pipline(date_range_data)

        self.train_model(x_train, y_train)
        
        primary_key_test = x_test.pop('primary_key').reset_index(drop=True)
        y_result = self.prediction_y(y_test, x_test, task_name='eval')
        
        prediction_stock_price = self.field_handle(y_result, x_test)
        
        # 通过主键关联字段
        related_name = ['date', 'code', 'code_name', 'preclose', 'isST']  # partition_date
        prediction_stock_price['primary_key'] = primary_key_test
        prediction_stock_price_related = pd.merge(prediction_stock_price, date_range_data[['primary_key']+related_name], on='primary_key')
        
        with base_connect_database.engine_conn('postgre') as conn:
            prediction_stock_price_related['insert_timestamp'] = datetime.now().strftime('%F %T')
            prediction_stock_price_related.to_sql(TRAIN_TABLE_NAME, con=conn.engine, index=False, if_exists='replace')
        
        return prediction_stock_price_related

    def train_model(self, x_train, y_train):
        """
        Train LightGBM model.
        """
        # feature
        model_metadata = {'primary_key_name': 'primary_key'}
        del x_train[model_metadata['primary_key_name']]
        model_metadata['feature_names'] = x_train.columns
        
        # fitting the model
        self.model_multioutput_regressor.fit(x_train, y_train)
        
        # save_model
        joblib.dump((self.model_multioutput_regressor, model_metadata), MULTIOUTPUT_MODEL_PATH)
        
# =============================================================================
#     def evaluate_x(self, lgb_train, params):
#         # 通过训练模型本身的参数输出对应的评估指标
#         # Use cross-validation to obtain evaluation results
#         cv_results = lgb.cv(params, lgb_train, nfold=5, metrics=params['metric'], verbose_eval=False)
# 
#         # Output the results
#         for metric in params['metric']:
#             avg_metric = f'average {metric} across folds'
#             print(f"{avg_metric}: {cv_results[f'{metric}-mean'][-1]:.2f} ± {cv_results[f'{metric}-stdv'][-1]:.2f}")
# 
#         return cv_results
# =============================================================================

    def prediction_y(self, y_test_true, x_test, task_name=None): #, prediction_name=None
        """
        evaluate_model
        Evaluate model performance, calculate RMSE and MAE, output results to the database, and return DataFrame.
        """
        ## pred
        y_test_pred = self.model_multioutput_regressor.predict(x_test)
        y_result = pd.DataFrame(np.hstack((y_test_true, y_test_pred)))
        
        ## eval
        mse = mean_squared_error(y_test_true, y_test_pred)
        rmse = mse ** (0.5)
        mae = mean_absolute_error(y_test_true, y_test_pred)
        
        insert_timestamp = datetime.now().strftime('%F %T')
        y_eval_dict = {'rmse': round(rmse,3),
                       'mae': round(mae,3),
                       'task_name': task_name,
                       'insert_timestamp': insert_timestamp,
                       }
        y_eval_df = pd.DataFrame([y_eval_dict], columns=y_eval_dict.keys())
        
        with base_connect_database.engine_conn('postgre') as conn:
            y_eval_df.to_sql(EVAL_TABLE_NAME, con=conn.engine, index=False, if_exists='append')
        return y_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date_start', type=str, default='2018-01-01', help='Start time for training')
    parser.add_argument('--date_end', type=str, default='2020-01-01', help='end time of training')
    args = parser.parse_args()

    print(f'Start time for training: {args.date_start}\nend time of training: {args.date_end}')
    
    # Load date range data
    #date_range_data = data_loading.feather_file_merge(args.date_start, args.date_end)
    with base_connect_database.engine_conn('postgre') as conn:
        date_range_data = pd.read_sql(f"SELECT * FROM history_a_stock_k_data WHERE date >= '{args.date_start}' AND date < '{args.date_end}'", con=conn.engine)
    
    stock_model = StockTrainModel()#date_start=args.date_start, date_end=args.date_end
    prediction_stock_price_related = stock_model.data_processing_pipline(date_range_data)
    
    # Rename and save to CSV file
    prediction_stock_price_related_rename = prediction_stock_price_related.rename(columns={'open': '今开盘价格',
                                                                                     'high': '最高价',
                                                                                     'low': '最低价',
                                                                                     'close': '今收盘价',
                                                                                     'volume': '成交数量',
                                                                                     'amount': '成交金额',
                                                                                     'turn': '换手率',
                                                                                     'pctChg': '涨跌幅',
                                                                                     'rear_low_real': '明天_最低价幅_真实值',
                                                                                     'rear_low_pred': '明天_最低价幅_预测值',
                                                                                     'rear_high_real': '明天_最高价幅_真实值',
                                                                                     'rear_high_pred': '明天_最高价幅_预测值',
                                                                                     'rear_diff_real': '明天_变化价幅_真实值',
                                                                                     'rear_diff_pred': '明天_变化价幅_预测值',
                                                                                     'remarks': '备注',
                                                                                     'date': '日期',
                                                                                    'code': '股票代码',
                                                                                    'code_name': '股票中文名称',
                                                                                    'isST': '是否ST',
                                                                                    })
    prediction_stock_price_related_rename.to_csv(PREDICTION_PRICE_OUTPUT_CSV_PATH, index=False)
    
    #stock_model.load_model(MODEL_PATH)

    # Plot feature importance
    #stock_model.plot_feature_importance()
    #stock_model.plot_feature_importance()





#w×RMSE+(1−w)×MAE

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
# =============================================================================
#     def plot_feature_importance(X_train):
#         """
#         Plot feature importance using Seaborn.
#     
#         Parameters:
#         - model_multioutput_regressor: The MultiOutputRegressor object.
#         - X_train: The training data used to fit the model.
#         """
#         # Assuming the underlying estimator is a LightGBM model
#         lgb_model = self.model_multioutput_regressor.estimators_[0].estimator_
#     
#         feature_importance_split = pd.DataFrame({
#             'Feature': X_train.columns,  # Assuming X_train is a DataFrame with named columns
#             'Importance': lgb_model.feature_importance(importance_type='split'),
#             'Type': 'split'
#         })
#     
#         feature_importance_gain = pd.DataFrame({
#             'Feature': X_train.columns,
#             'Importance': lgb_model.feature_importance(importance_type='gain'),
#             'Type': 'gain'
#         })
#     
#         feature_importance = pd.concat([feature_importance_split, feature_importance_gain], ignore_index=True)
#     
#         plt.figure(figsize=(12, 6))
#         sns.barplot(x='Importance', y='Feature', data=feature_importance, hue='Type', palette="viridis", dodge=True)
#         plt.title('Feature Importance')
#         plt.show()
# =============================================================================
# =============================================================================
#     def plot_feature_importance(self):
#         """
#         Plot feature importance using Seaborn.
#         """
#         feature_importance_split = pd.DataFrame({
#             'Feature': self.model_multioutput_regressor.feature_name(),
#             'Importance': self.model_multioutput_regressor.feature_importance(importance_type='split'),
#             'Type': 'split'
#         })
#         
#         feature_importance_gain = pd.DataFrame({
#             'Feature': self.model_multioutput_regressor.feature_name(),
#             'Importance': self.model_multioutput_regressor.feature_importance(importance_type='gain'),
#             'Type': 'gain'
#         })
# 
#         feature_importance = pd.concat([feature_importance_split, feature_importance_gain], ignore_index=True)
# 
#         plt.figure(figsize=(12, 6))
#         sns.barplot(x='Importance', y='Feature', data=feature_importance, hue='Type', palette="viridis", dodge=True)
#         plt.title('Feature Importance')
#         plt.show()
#         
# =============================================================================
# =============================================================================
#     def plot_feature_importance(self):
#         """
#         Plot feature importance.
#         """
#         lgb.plot_importance(self.model, importance_type='split', figsize=(10, 6), title='Feature importance (split)')
#         lgb.plot_importance(self.model, importance_type='gain', figsize=(10, 6), title='Feature importance (gain)')
# =============================================================================