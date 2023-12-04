# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 10:58:16 2023

@author: awei
应用层_每日定时任务(application_daily)
"""
import argparse
import schedule
from datetime import datetime, timedelta

import baostock as bs

from __init__ import path
from base import base_connect_database, base_arguments
from get_data import data_plate

def custom_date(date=None):
    """
    功能：直接执行当日的每日任务
    """
    try:
        conn_pg = base_connect_database.engine_conn('postgre')
        lg = bs.login()  # 登陆系统
        
        data_plate.get_base_data(data_type='交易日', conn=conn_pg.engine)
        data_plate.get_base_data(data_type='行业分类', conn=conn_pg.engine)
        data_plate.get_base_data(data_type='证券资料', conn=conn_pg.engine)
        data_plate.get_base_data(data_type='证券代码', conn=conn_pg.engine)
    except Exception as e:
        print(f"登录获取交易日数据异常: {e}")
    finally:
        bs.logout()  # 登出系统

def time():
    """
    功能：任务定时
    """
    schedule.every().day.at("01:00").do(custom_date)
    while True:
        schedule.run_pending()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default='now')  # 默认daily每日离线任务,可选['daily','now','2022-09-13']
    args = parser.parse_args()
    
    if args.date not in ['daily', 'now']:
        custom_date(args.date)
    elif args.date == 'now':
        custom_date()
    elif args.date == 'daily':
        time()




