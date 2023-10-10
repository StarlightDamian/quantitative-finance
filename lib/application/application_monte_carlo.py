# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 17:38:05 2023

@author: awei
应用层_蒙特卡洛方法_模拟交易
application_monte_carlo
"""
import pandas as pd
#from base import base_data_loading
from __init__ import path

if __name__ == '__main__':
    data_df = pd.read_csv(f'{path}/data/stock_industry.csv', encoding='gb18030')
    plate_df = data_df[data_df.industry.isin(['计算机'])]