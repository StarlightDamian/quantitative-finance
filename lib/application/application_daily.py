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
from get_data import data_plate, data_history_a_stock_k_data

def custom_date(date=None):
    """
    功能：Directly execute daily tasks for the day
    """
    try:
        conn_pg = base_connect_database.engine_conn('postgre')
        lg = bs.login()
        
        data_plate.get_base_data(data_type='交易日', conn=conn_pg.engine)
        data_plate.get_base_data(data_type='行业分类', conn=conn_pg.engine)
        data_plate.get_base_data(data_type='证券资料', conn=conn_pg.engine)
        data_plate.get_base_data(data_type='证券代码', conn=conn_pg.engine)
    except Exception as e:
        print(f"Exception when logging in to obtain transaction day data: {e}")
    finally:
        bs.logout()

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




