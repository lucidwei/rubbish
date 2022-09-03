# coding=gbk
# lucid   当前系统用户
# 2022/8/16   当前系统日期
# 14:47   当前系统时间
# PyCharm   创建文件的IDE名称
import datetime
import warnings, os.path
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

# 针对wind excel数据复用
# minor warning: wind下载的自定义合成指标没有完整的指标ID，列名有极小的概率重叠，其或造成feature丢失。8月19日版本数据暂无此问题。
class GetStructuralData:
    # 对象存储数据（字典类型）和描述表，类方法功能包括1格式化（取值变更坐标）2将数据和描述表存入data文件夹
    def __init__(self, path_read, update=False):
        print('update:', update)
        print('os.path.exists(csv):', os.path.exists(r'data/data_x.csv'))
        self.path_read = path_read
        self.update = update
        if update and os.path.exists(r'data/data_x.csv'):
            self.raw_data = self.read_files()
            self.info = self.get_info(last_info_row=6)
            self.structural_data = self.get_sdata(last_info_row=6)
        else:
            self.info = pd.read_csv(r'data/info_table.csv', index_col=0, parse_dates=True)
            self.structural_data = {'x': pd.read_csv(r'data/data_x.csv', index_col=0, parse_dates=True),
                                    'y': pd.read_csv(r'data/data_y.csv', index_col=0, parse_dates=True)}

    def read_files(self):
        md_dic = pd.read_excel(self.path_read + r'\国内宏观.xlsx', engine="openpyxl", sheet_name=None)
        mo = pd.read_excel(self.path_read + r'\海外宏观.xlsx', engine="openpyxl")
        da = pd.read_excel(self.path_read + r'\日频大类资产.xlsx', engine="openpyxl")
        di = pd.read_excel(self.path_read + r'\日频衍生指标.xlsx', engine="openpyxl")
        dic = {'macro_domestic': md_dic, 'macro_overseas': mo, 'daily_assets': da, 'daily_indicators': di}
        return dic

    def get_info(self, last_info_row):
        # TODO: 优先级低 指标类别&发布时点（先省略了 后续优化时加入）。
        # 每行一个指标，columns包括指标ID，指标名称。备用包括频率，单位，来源，国家，更新时间。
        def by_tablename(tbname: str):
            info_by_tb = self.raw_data[tbname][:last_info_row + 1].T
            info_by_tb.set_axis(info_by_tb.iloc[:, 1], axis='index', inplace=True)
            info_by_tb.set_axis(info_by_tb.iloc[0, :], axis='columns', inplace=True)
            info_by_tb.drop('指标ID', axis=0, inplace=True)
            info_by_tb.drop('指标ID', axis=1, inplace=True)
            return info_by_tb

        def by_dicname(dicname: str):
            sheets = self.raw_data[dicname].keys()
            info_by_dic = None
            for i in sheets:
                # TODO: 优先级低 高端操作，可以练习wrapper
                info_by_tb = self.raw_data[dicname][i][:last_info_row + 1].T
                info_by_tb.set_axis(info_by_tb.iloc[:, 1], axis='index', inplace=True)
                info_by_tb.set_axis(info_by_tb.iloc[0, :], axis='columns', inplace=True)
                info_by_tb.drop('指标ID', axis=0, inplace=True)
                info_by_tb.drop('指标ID', axis=1, inplace=True)
                info_by_tb.insert(1, '指标类别', dicname + i)
                info_by_dic = pd.concat([info_by_dic, info_by_tb])
            return info_by_dic

        info_md = by_dicname('macro_domestic')
        info_mo = by_tablename('macro_overseas')
        info_di = by_tablename('daily_indicators')
        info_da = by_tablename('daily_assets')
        info = pd.concat([info_md, info_mo, info_di, info_da])
        if not os.path.exists(r'data/info_table.csv') or self.update:
            info.to_csv(r'.\data\info_table.csv')
        return info

    def get_sdata(self, last_info_row):
        def by_tablename(tbname: str):
            data_by_tb = self.raw_data[tbname]
            col = data_by_tb.iloc[1, :]
            col[0] = '日期'
            data_by_tb.set_axis(col, axis='columns', inplace=True)
            # 最后两行是无效信息
            data_by_tb = data_by_tb[last_info_row + 1:-2].copy()
            data_by_tb.loc[:, '日期'] = pd.DatetimeIndex(data_by_tb['日期'])
            data_by_tb.set_index('日期', inplace=True)
            return data_by_tb

        def by_dicname(dicname: str):
            sheets = self.raw_data[dicname].keys()
            data_by_dic = None
            for i in sheets:
                # TODO: 优先级低 练习wrapper
                data_by_tb = self.raw_data[dicname][i]
                col = data_by_tb.iloc[1, :]
                col[0] = '日期'
                data_by_tb.set_axis(col, axis='columns', inplace=True)
                data_by_tb = data_by_tb[last_info_row + 1:-2].copy()
                # 便于后续resample
                data_by_tb.loc[:, '日期'] = pd.DatetimeIndex(data_by_tb['日期'])
                data_by_tb.set_index('日期', inplace=True)
                data_by_dic = pd.concat([data_by_dic, data_by_tb], axis=1)
            return data_by_dic

        data_md = by_dicname('macro_domestic')
        data_mo = by_tablename('macro_overseas')
        data_di = by_tablename('daily_indicators')
        sdata_y = by_tablename('daily_assets')
        sdata_x = pd.concat([data_md, data_mo, data_di], axis=1)
        if not os.path.exists(r'data/data_x.csv') or self.update:
            sdata_x.to_csv(r'.\data\data_x.csv')
            sdata_y.to_csv(r'.\data\data_y.csv')
        return {'x': sdata_x, 'y': sdata_y}


# def fill_na_naive(df):
#     df.fillna(method='ffill', inplace=True)
#     df.fillna(method='bfill', inplace=True)
#     return df
#
#
# # TODO: 低优先级 单拎出来，后续若详细优化数据获取时间需要在这里下手。
# def fill_x_na(df, info_df):
#     # 个性化操作高频df，注：对于起始统计晚的数据并没有进行bfill
#     for id in df.columns:
#         if info_df.loc[id, 'fillna'] == 'ffill':
#             df.loc[:, id].fillna(method='ffill', inplace=True)
#         elif info_df.loc[id, 'fillna'] == '0fill':
#             df.loc[:, id].fillna(value=0, inplace=True)
#         elif pd.isnull(info_df.loc[id, 'fillna']):
#             pass
#         else:
#             raise Exception('donno how to fillna')
#     return df
#
#
# def station_test(ts):
#     def kpss_test(timeseries):
#         # print("Results of KPSS Test:")
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             kpsstest = kpss(timeseries, regression="c", nlags="auto")
#         kpss_output = pd.Series(
#             kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
#         )
#         # print(kpss_output)
#         return kpss_output[1]
#
#     def adf_test(timeseries):
#         # print("Results of Dickey-Fuller Test:")
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             dftest = adfuller(timeseries, autolag="AIC")
#         dfoutput = pd.Series(
#             dftest[0:4],
#             index=[
#                 "Test Statistic",
#                 "p-value",
#                 "#Lags Used",
#                 "Number of Observations Used",
#             ], )
#         # print(dfoutput)
#         return dfoutput[1]
#
#     p_adf = adf_test(ts)
#     p_kpss = kpss_test(ts)
#     threshold = 0.05
#     if p_adf <= threshold <= p_kpss:
#         return 'stationary'
#     elif p_kpss < threshold < p_adf:
#         return 'non-stationary'
#     elif p_adf > threshold and p_kpss > threshold:
#         return 'trend-stationary'
#     elif p_adf < threshold and p_kpss < threshold:
#         return 'diff-stationary'
#     else:
#         raise Exception('donno stationarity')
#
#
# class Preprocess:
#     def __init__(self, s_data: dict, align_to='month', begT='2005-01', endT=datetime.date.today(), lag_month=1):
#         self.s_datax = s_data['x'][begT:endT]
#         self.s_datay = s_data['y'].iloc[::-1][begT:endT]
#         self.info = pd.read_excel(r'.\data\to_month_table.xlsx', index_col=0,
#                                   engine="openpyxl") if align_to == 'month' else \
#             pd.read_excel(r'.\data\to_week_table.xlsx', index_col=0, engine="openpyxl")
#         self.special_treatment()
#         # 时间对齐
#         self.data_aligned = {}
#         if align_to == 'month':
#             self.align_to_month()
#         elif align_to == 'week':
#             # TODO: 低优先级 周度align规则尚需手动完成
#             self.align_to_week()
#         else:
#             raise Exception('donno how to align data')
#         # 高阶预处理 有一次fillna
#         self.get_stationary()
#         # 转换成监督学习格式
#         self.lag = lag_month
#         self.data_dic = self.series_to_supervised(2, 2)
#         pass
#
#     # test and then detrend and/or seasonal adjust
#     def get_stationary(self):
#         # TODO: 低优先级 先简易fillna(不fillna不能stl)，有些规律性空值的或许有计算差错（未验证猜想）
#         df = fill_na_naive(self.data_aligned['x'])
#         record = {col_ind: station_test(col) for col_ind, col in df.iteritems()}
#         # count = pd.value_counts(list(record.values()))
#         # print('There are %d not stationary variables' % (len(record) - count['stationary']))
#         # 不知道trend有没有用，trend和residual都留着吧
#         for col_ind, col in df.iteritems():
#             if record[col_ind] != 'stationary':
#                 stl = STL(col, period=12)
#                 decomposed = stl.fit()
#                 df[col_ind + '_trend'] = decomposed.trend
#                 df[col_ind + '_resid'] = decomposed.resid
#                 df.drop(col_ind, inplace=True, axis=1)
#         self.data_aligned['x'] = df
#
#     def get_datadic(self):
#         # 一些算法(如RF)不支持空值
#         return self.data_dic
#
#     # TODO: 中优先级 实现自定义lag
#     #  低优先级 模拟生产环境 比较复杂 针对数据发布时间优化，没发布之前统一用之前的数据，另外函数应该做到可以复用到周度数据
#     def series_to_supervised(self, n_in=1, n_out=1, dropnan=True):
#         # 默认Y不变，X向未来移一个月并保留，相当于用上个月的X来预测这个月的Y。可以模拟这个月才得到上个月的宏观数据。
#         # 如果需要增加时移feature可调大n_in，同理n_out。
#         dfs, col_names = list(), list()
#         data_dic = {}
#         # input sequence (t-n, ... t-1)
#         x_df = self.data_aligned['x'].copy()
#         for i in range(n_in - 1, 0, -1):
#             dfs.append(x_df.shift(i))
#             col_names += [('var%d(t-%d)' % (j + 1, i) + x_df.columns.values[j]) for j in range(x_df.shape[1])]
#         data_dic['x'] = pd.concat(dfs, axis=1)
#         data_dic['x'].columns = col_names
#         # forecast sequence (t, t+1, ... t+n)
#         dfs, col_names = list(), list()
#         y_df = self.data_aligned['y'].copy()
#         for i in range(0, n_out - 1):
#             dfs.append(y_df.shift(-i))
#             if i == 0:
#                 col_names += [('var%d(t)' % (j + 1) + y_df.columns.values[j]) for j in range(y_df.shape[1])]
#             else:
#                 col_names += [('var%d(t+%d)' % (j + 1, i) + y_df.columns.values[j]) for j in range(y_df.shape[1])]
#         data_dic['y'] = pd.concat(dfs, axis=1)
#         data_dic['y'].columns = col_names
#
#         if dropnan:
#             # drop rows with NaN values
#             data_dic['x'].dropna(inplace=True)
#             data_dic['y'].dropna(inplace=True)
#             # 只保留x y共有的日期
#             common_ind = data_dic['x'].index.intersection(data_dic['y'].index)
#             data_dic['x'] = data_dic['x'].loc[common_ind]
#             data_dic['y'] = data_dic['y'].loc[common_ind]
#
#         return data_dic
#
#     # TODO: 低优先级 单拎出来，后续若详细优化数据获取时间需要在这里下手。
#     # def fill_x_na(self):
#     #     # 个性化操作高频df，注：对于起始统计晚的数据并没有进行bfill
#     #     df = self.s_datax.copy()
#     #     for id in df.columns:
#     #         if self.info.loc[id, 'fillna'] == 'ffill':
#     #             df.loc[:, id].fillna(method='ffill', inplace=True)
#     #         elif self.info.loc[id, 'fillna'] == '0fill':
#     #             df.loc[:, id].fillna(value=0, inplace=True)
#     #         elif pd.isnull(self.info.loc[id, 'fillna']):
#     #             pass
#     #         else:
#     #             raise Exception('donno how to fillna')
#     #     return df
#
#     def align_to_month(self):
#         # 先fillna再resample
#         def get_y():
#             self.s_datay.fillna(method='ffill', inplace=True)
#             # 对于一些交易数据起始时间晚的品种需要补空
#             self.s_datay.fillna(method='bfill', inplace=True)
#             df = deepcopy(self.s_datay)
#             # 获取数据中存在的月末日期：根据辅助列year和month进行groupby,然后取每个分组中最大（max）的日期，并存放到month_end列表中。
#             # 有些冗余，当初想保留月末日期，后来发现和x月末日期对不上，故只保留月。
#             df['date'] = df.index
#             df['year'] = df['date'].apply(lambda x: x.year)
#             df['month'] = df['date'].apply(lambda x: x.month)
#             grouped = df['date'].groupby([df['year'], df['month']])
#             month_end = grouped.max().to_list()
#             df.drop(['year', 'month', 'date'], axis=1, inplace=True)
#             # 转换成收益率，最开始第一个月末没有收益率，最后一个月不是完整月收益
#             month_end_df = df.loc[month_end, :].pct_change()
#             # index从月末交易日对齐到月份
#             month_end_df.index = month_end_df.index.to_period('M')
#             return month_end_df
#
#         # 针对x fillna和resample
#         def get_x():
#             # 插入资产走势
#             df = pd.concat([self.s_datax, self.s_datay.copy()], axis=1)
#             df = fill_x_na(df, self.info)
#             # 可以groupby后直接resample
#             month_end_df = pd.DataFrame()
#             for id in df.columns:
#                 if self.info.loc[id, 'resample(月)'] == 'last':
#                     ts = df.loc[:, id].resample('1M').last()
#                 elif self.info.loc[id, 'resample(月)'] == 'avg':
#                     ts = df.loc[:, id].resample('1M').mean()
#                 elif self.info.loc[id, 'resample(月)'] == 'cumsum':
#                     # 备注一下数据被transform了
#                     ts = df.loc[:, id].cumsum().resample('1M').last()
#                     self.info.loc[id, '指标名称'] = 'cumsumed' + self.info.loc[id, '指标名称']
#                 else:  # 剩下的应该是资产走势，直接取月末
#                     ts = df.loc[:, id].resample('1M').last()
#                 month_end_df = pd.concat([month_end_df, ts], axis=1)
#             # index从月末交易日对齐到月份
#             # noinspection PyTypeChecker
#             month_end_df.index = pd.to_datetime(month_end_df.index).to_period('M')
#             return fill_na_naive(month_end_df)
#
#         # Y的始末月不能用，X对齐
#         y = get_y()[1: -1]
#         x = get_x()[1: -1]
#         # 插入资产收益
#         y_return = deepcopy(y)
#         y_return.columns = y_return.columns.map(lambda x: x+'_returns')
#         x = pd.concat([x, y_return], axis=1)
#
#         self.data_aligned = {'x': x, 'y': y}
#
#     def align_to_week(self):
#         self.data_aligned = {'x': self.s_datax.resample('1w').ffill(),
#                              'y': self.s_datay.resample('1w').ffill()}
#         pass
#
#     def special_treatment(self):
#         # MLF利率 M5528820(旧、删除该列)拼接到M0329545(新)
#         if 'M5528820' in self.s_datax.columns:
#             self.s_datax.loc[:, 'M0329545'] = self.s_datax.M5528820.copy().add(self.s_datax.M0329545.copy(),
#                                                                                fill_value=0).copy()
#             self.s_datax.drop('M5528820', axis=1, inplace=True)
#         # 累计值转化为当月值
#         for id in self.s_datax.columns:
#             if self.info.loc[id, '是否累计值'] == '是':
#                 self.info.loc[id, '指标名称'] = '(月度化)' + self.info.loc[id, '指标名称']
#                 sr = self.s_datax[id]
#                 sr_ori = deepcopy(sr)
#                 for date in sr.index:
#                     if np.isfinite(sr[date]):
#                         try:
#                             diff = sr[date] - sr_ori[date - pd.offsets.MonthEnd()]
#                             self.s_datax.loc[date, id] = diff if diff > 0 else sr[date]
#                         except:
#                             pass
#         # 去掉一致预期数据？先仅仅掐掉未来数据吧
#         pass
#
#
# class FeatureEngineer:
#     pass
#
#
# class MacroModeling:
#     def __init__(self, data_dic, train_pct=0.7):
#         self.data_dic = data_dic
#         self.train_dic, self.test_dic = self.get_split_data(train_pct)
#         self.trained = self.train()
#         self.predict()
#
#     # 划分出总的训练（划分训练子和验证用tscv）和测试集。
#     def get_split_data(self, train_pct):
#         if len(self.data_dic['x']) != len(self.data_dic['y']):
#             raise Exception('数据中x y序列不等长')
#         length = int(len(self.data_dic['x']) * train_pct)
#         train_x = self.data_dic['x'].iloc[0:length]
#         train_y = self.data_dic['y'].iloc[0:length]
#         test_x = self.data_dic['x'].iloc[length:]
#         test_y = self.data_dic['y'].iloc[length:]
#         return {'x': train_x, 'y': train_y}, {'x': test_x, 'y': test_y}
#
#     # in-sample
#     def train(self):
#         # TimeSeriesSplit中的gap是干嘛的？应该是跟统计学原理有关。不知是否重要。
#
#         tscv = TimeSeriesSplit()
#         rfgs_parameters = {
#             'n_estimators': [n for n in range(30, 35)],
#             'max_depth': [n for n in range(2, 5)],
#             'max_features': [n for n in range(2, 5)],
#             "min_samples_split": [n for n in range(2, 5)],
#             "min_samples_leaf": [n for n in range(2, 5)],
#             "bootstrap": [True, False]
#         }
#
#         # TODO: MultiOutput和直接regress多个output有何区别？
#         rfr_cv = GridSearchCV(RandomForestRegressor(), rfgs_parameters, cv=tscv, scoring='r2')
#
#         print('start training')
#         rfr_cv.fit(self.train_dic['x'], self.train_dic['y'])
#         print("RFR GridSearch score: " + str(rfr_cv.best_score_))
#         print("RFR GridSearch params: ")
#         print(rfr_cv.best_params_)
#
#         return rfr_cv
#
#     # out-sample
#     def predict(self):
#         from sklearn.metrics import r2_score
#         prediction1 = self.trained.best_estimator_.predict(self.test_dic['x'])
#         # print(prediction1)
#         print('r2_score:', r2_score(self.test_dic['y'], prediction1))
#
#     pass
#
#
# class Evaluation:
#     pass
