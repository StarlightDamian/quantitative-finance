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
from sqlalchemy import create_engine
from datetime import datetime, timedelta

import baostock as bs

# from __init__ import path
from get_data import data_loading
from base import base_connect_database


def get_base_data(data_type, conn):
    """
    获取常规数据
    :param data_type:获取数据类型
    :param conn:连接方式
    备注：1.更新频率：天
    """
    if data_type == '交易日':  # 获取交易日
        rs = bs.query_trade_dates()
        filename = 'trade_datas'
        
    elif data_type == '行业分类':  # 获取行业分类数据
        rs = bs.query_stock_industry()
        filename = 'stock_industry'
        
    elif data_type == '证券资料':  # 获取证券基本资料
        rs = bs.query_stock_basic()
        filename = 'stock_basic'
        
    elif data_type == '证券代码':  # 获取证券代码
        date = (datetime.now()+timedelta(days=-1)).strftime('%F')  # 取当天的不一定及时更新，先尝试前一天
        rs = bs.query_all_stock(date)
        filename = 'all_stock'
        
    result = data_loading.re_get_row_data(rs)
    #result.to_csv(f"{path}/data/{filename}.csv", encoding="gbk", index=False)
    now = datetime.now().strftime("%F %T")
    result['last_updated'] = now
    result.to_sql(filename, con=conn, index=False, if_exists='replace')  # 输出到数据库
    print(f'数据成功入库： {data_type}\n当前时间： {now}\n==============')
    
    # return result


if __name__ == '__main__':
    conn_pg = base_connect_database.engine_conn('postgre')
    lg = bs.login()  # 登陆系统
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, default='交易日', help='["交易日", "行业分类", "证券资料", "证券代码"]')
    args = parser.parse_args()
    
    result = get_base_data(args.data_type, conn=conn_pg)
    print(result)
    
    bs.logout()  # 登出系统
    