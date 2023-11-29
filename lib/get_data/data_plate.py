# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 15:01:44 2023

@author: awei
data_plate
获取证券元信息
1交易日
2获取股票对应的板块
3获取证券基本资料
4获取证券代码

"""
import argparse
from datetime import datetime, timedelta

import baostock as bs

from __init__ import path
from get_data import data_loading


def get_base_data(data_type):
    if data_type == '交易日':  # 获取交易日
        rs = bs.query_trade_dates()
        result = data_loading.re_get_row_data(rs)
        result.to_csv(f"{path}/data/trade_datas.csv", encoding="gbk", index=False)
    elif data_type == '行业分类':  # 获取行业分类数据
        rs = bs.query_stock_industry()
        result = data_loading.re_get_row_data(rs)
        result.to_csv(f"{path}/data/stock_industry.csv", encoding="gbk", index=False)
    elif args.data_type == '证券资料':  # 获取证券基本资料
        rs = bs.query_stock_basic()
        result = data_loading.re_get_row_data(rs)
        result.to_csv(f"{path}/data/stock_basic.csv", encoding="gbk", index=False)
    elif data_type == '证券代码':  # 获取证券代码
        date = (datetime.now()+timedelta(days=-1)).strftime('%F')  # 取当天的不一定及时更新，先尝试前一天
        rs = bs.query_all_stock(date)
        result = data_loading.re_get_row_data(rs)
        result.to_csv(f"{path}/data/all_stock.csv", encoding="gbk", index=False)

    return result


if __name__ == '__main__':
    lg = bs.login()  # 登陆系统
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='交易日', help='["交易日", "行业分类", "证券资料", "证券代码"]')
    args = parser.parse_args()
    
    result = get_base_data(args.data_type)
    print(result)
    
    bs.logout()  # 登出系统
    