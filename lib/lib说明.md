## lib说明

|          类别           |          python文件           |              说明               |                            备注                             |
| :---------------------: | :---------------------------: | :-----------------------------: | :---------------------------------------------------------: |
|     **application**     |                               |           应用文件夹            |  在这一层通过控制变量来对指定金额、策略、历史数据进行回测   |
|     **backtesting**     |                               |         量化回测文件夹          |                                                             |
|        **base**         |                               |           基础文件夹            |                      工具包和数据获取                       |
|          base           |     **base_data_loading**     |            数据加载             |                                                             |
|          base           |     **base_handling_fee**     |           手续费计算            |                                                             |
|          base           |        **base_utils**         |             工具包              |                                                             |
|          base           |         **base_relu**         |            交易规则             | ETF交易没限制，普通股票>100股<br />周末不交易、节假日不交易 |
|          base           |   **base_risk_management**    |            风险管理             |                  评估股票、交易的风险程度                   |
|          base           |      **base_portfolio**       |            投资组合             |            通过投资组合来获得更稳定的收益和复利             |
| **feature_engineering** |                               |         特征工程文件夹          |                                                             |
|      **get_data**       |                               |         获取数据文件夹          |                                                             |
|        get_data         |         **data_get**          |            获取数据             |                                                             |
|        get_data         |     **data_distribution**     |            数据分发             |                                                             |
|        get_data         | **data_get_one_day_all_data** | 获取指定日期全部股票的日K线数据 |                                                             |
|        get_data         |       **data_loading**        |            数据加载             |                                                             |
|        get_data         |        **data_plate**         |       获取股票对应的板块        |                                                             |
|      **strategy**       |                               |         量化策略文件夹          |                                                             |
|        **train**        |                               |           训练文件夹            |                                                             |
|    **visualization**    |                               |          可视化文件夹           |                                                             |
|                         |                               |                                 |                                                             |



