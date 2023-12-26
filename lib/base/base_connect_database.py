# -*- coding: utf-8 -*-
"""
Created on Fri Feb 18 14:53:45 2022

@author: admin
连接数据库(base_connect_database)

数据库连接模块考虑情况：
1.数据库连接池，连接数量
2.engine连接，普通连接，他们读写数据库表的写法会有差异
3.考虑windows和linux连接差异，PooledDB包在不同系统下的读取方式不同。连接hive在windows环境需要impala
4.有时候项目只需要部分类型的包支持指定数据库，如只安装hive和postgre相应的包
5.不同类型连接的参数不一致，比如hive有'auth'、'auth_mechanism'
6.支持windows和Linux连接同一类数据库，但是host：post不一致。windows测试，Linux正式
"""
from urllib import parse
from sqlalchemy import create_engine

import pandas as pd

from __init__ import path
from base import base_arguments as arg


class DatabaseConnection:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)

    def __enter__(self):
        self.conn = self.engine.connect()
        return self.conn

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            self.conn.close()


def engine_conn(type_database):
    """
    功能：连接数据库
    备注：输出至数据库：to_csv()  if_exists:['append','replace','fail']#追加、删除原表后新增、啥都不干抛出一个 ValueError
    """
    #print(f"当前数据库：{type_database}")
    user = arg.conf(f'{type_database}_user')
    password = arg.conf(f'{type_database}_password')
    password = parse.quote_plus(str(password))  # 处理密码中带有@，被create_engine误分割导致的BUG
    host = arg.conf(f'{type_database}_host')
    port = arg.conf(f'{type_database}_port')
    database = arg.conf(f'{type_database}_database')
    database_dict = {'hive': 'hive', 'postgre': 'postgresql', 'oracle': 'oracle', 'mysql': 'mysql+pymysql'}
    database_name = database_dict.get(f"{type_database}")
    user_password_host_port_database_str = f"{user}:{password}@{host}:{port}/{database}"

    if type_database == 'hive':
        auth = arg.conf('hive_auth')
        db_url = f"{database_name}://{user_password_host_port_database_str}?auth={auth}"
    elif type_database in ['postgre', 'oracle', 'mysql']:
        db_url = f"{database_name}://{user_password_host_port_database_str}"

    return DatabaseConnection(db_url)

if __name__ == '__main__':
    # print(path)
    
    # Data exploration
    with engine_conn('postgre') as conn:
        table_name = 'history_a_stock_k_data'
        count_field = 'date'
        #data = pd.read_sql("SELECT * FROM history_a_stock_k_data limit 10", con=conn.engine)  # Use conn_pg.engine
        data = pd.read_sql(f"SELECT {count_field}, COUNT(*) AS data_count FROM {table_name} GROUP BY {count_field} ORDER BY {count_field}", con=conn.engine)
        print(data)
    
    print('The amount of data:', data.data_count.sum())
    data.to_csv(f'{path}/data/history_a_stock_k_data_count_date.csv')
    
    #from base import base_utils
    #data['primaryKey'] = (data['date']+data['code']).apply(base_utils.md5_str) # md5（日期、时间、代码）
    #from sqlalchemy import Float, Numeric, String
# =============================================================================
#     data.to_sql('history_a_stock_k_data', con=conn.engine, index=False, if_exists='replace',
#                             dtype={
#                                 'primaryKey': String,
#                                 'date': String,
#                                 'code': String,
#                                 'code_name': String,
#                                 'open': Float,
#                                 'high': Float,
#                                 'low': Float,
#                                 'close': Float,
#                                 'preclose': Float,
#                                 'volume': Numeric,
#                                 'amount': Numeric,
#                                 'adjustflag': String,
#                                 'turn': Float,
#                                 'tradestatus': String,
#                                 'pctChg': Float,
#                                 'peTTM': Float,
#                                 'psTTM': Float,
#                                 'pcfNcfTTM': Float,
#                                 'pbMRQ': Float,
#                                 'isST': String,
#                             })
# =============================================================================
    # conn_pg = engine_conn('postgre')
    # data = pd.read_sql("SELECT * FROM warning_hot_word_mx limit 10",con=conn_pg)
    
    #writer = pd.ExcelWriter(f'{path}/data/chongfuzisha_jjd_20220401_20220715.xlsx')
    #data.to_excel(writer, sheet_name='20220401_20220714重复自杀接警单', index=False)
    #writer.save()
