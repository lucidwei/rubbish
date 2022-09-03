# coding=gbk
# lucid   ��ǰϵͳ�û�
# 2022/8/16   ��ǰϵͳ����
# 14:47   ��ǰϵͳʱ��
# PyCharm   �����ļ���IDE����
import datetime
import warnings, os.path
from copy import deepcopy

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

# ���wind excel���ݸ���
# minor warning: wind���ص��Զ���ϳ�ָ��û��������ָ��ID�������м�С�ĸ����ص���������feature��ʧ��8��19�հ汾�������޴����⡣
class GetStructuralData:
    # ����洢���ݣ��ֵ����ͣ����������෽�����ܰ���1��ʽ����ȡֵ������꣩2�����ݺ����������data�ļ���
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
        md_dic = pd.read_excel(self.path_read + r'\���ں��.xlsx', engine="openpyxl", sheet_name=None)
        mo = pd.read_excel(self.path_read + r'\������.xlsx', engine="openpyxl")
        da = pd.read_excel(self.path_read + r'\��Ƶ�����ʲ�.xlsx', engine="openpyxl")
        di = pd.read_excel(self.path_read + r'\��Ƶ����ָ��.xlsx', engine="openpyxl")
        dic = {'macro_domestic': md_dic, 'macro_overseas': mo, 'daily_assets': da, 'daily_indicators': di}
        return dic

    def get_info(self, last_info_row):
        # TODO: ���ȼ��� ָ�����&����ʱ�㣨��ʡ���� �����Ż�ʱ���룩��
        # ÿ��һ��ָ�꣬columns����ָ��ID��ָ�����ơ����ð���Ƶ�ʣ���λ����Դ�����ң�����ʱ�䡣
        def by_tablename(tbname: str):
            info_by_tb = self.raw_data[tbname][:last_info_row + 1].T
            info_by_tb.set_axis(info_by_tb.iloc[:, 1], axis='index', inplace=True)
            info_by_tb.set_axis(info_by_tb.iloc[0, :], axis='columns', inplace=True)
            info_by_tb.drop('ָ��ID', axis=0, inplace=True)
            info_by_tb.drop('ָ��ID', axis=1, inplace=True)
            return info_by_tb

        def by_dicname(dicname: str):
            sheets = self.raw_data[dicname].keys()
            info_by_dic = None
            for i in sheets:
                # TODO: ���ȼ��� �߶˲�����������ϰwrapper
                info_by_tb = self.raw_data[dicname][i][:last_info_row + 1].T
                info_by_tb.set_axis(info_by_tb.iloc[:, 1], axis='index', inplace=True)
                info_by_tb.set_axis(info_by_tb.iloc[0, :], axis='columns', inplace=True)
                info_by_tb.drop('ָ��ID', axis=0, inplace=True)
                info_by_tb.drop('ָ��ID', axis=1, inplace=True)
                info_by_tb.insert(1, 'ָ�����', dicname + i)
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
            col[0] = '����'
            data_by_tb.set_axis(col, axis='columns', inplace=True)
            # �����������Ч��Ϣ
            data_by_tb = data_by_tb[last_info_row + 1:-2].copy()
            data_by_tb.loc[:, '����'] = pd.DatetimeIndex(data_by_tb['����'])
            data_by_tb.set_index('����', inplace=True)
            return data_by_tb

        def by_dicname(dicname: str):
            sheets = self.raw_data[dicname].keys()
            data_by_dic = None
            for i in sheets:
                # TODO: ���ȼ��� ��ϰwrapper
                data_by_tb = self.raw_data[dicname][i]
                col = data_by_tb.iloc[1, :]
                col[0] = '����'
                data_by_tb.set_axis(col, axis='columns', inplace=True)
                data_by_tb = data_by_tb[last_info_row + 1:-2].copy()
                # ���ں���resample
                data_by_tb.loc[:, '����'] = pd.DatetimeIndex(data_by_tb['����'])
                data_by_tb.set_index('����', inplace=True)
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
# # TODO: �����ȼ� �����������������ϸ�Ż����ݻ�ȡʱ����Ҫ���������֡�
# def fill_x_na(df, info_df):
#     # ���Ի�������Ƶdf��ע��������ʼͳ��������ݲ�û�н���bfill
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
#         # ʱ�����
#         self.data_aligned = {}
#         if align_to == 'month':
#             self.align_to_month()
#         elif align_to == 'week':
#             # TODO: �����ȼ� �ܶ�align���������ֶ����
#             self.align_to_week()
#         else:
#             raise Exception('donno how to align data')
#         # �߽�Ԥ���� ��һ��fillna
#         self.get_stationary()
#         # ת���ɼලѧϰ��ʽ
#         self.lag = lag_month
#         self.data_dic = self.series_to_supervised(2, 2)
#         pass
#
#     # test and then detrend and/or seasonal adjust
#     def get_stationary(self):
#         # TODO: �����ȼ� �ȼ���fillna(��fillna����stl)����Щ�����Կ�ֵ�Ļ����м�����δ��֤���룩
#         df = fill_na_naive(self.data_aligned['x'])
#         record = {col_ind: station_test(col) for col_ind, col in df.iteritems()}
#         # count = pd.value_counts(list(record.values()))
#         # print('There are %d not stationary variables' % (len(record) - count['stationary']))
#         # ��֪��trend��û���ã�trend��residual�����Ű�
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
#         # һЩ�㷨(��RF)��֧�ֿ�ֵ
#         return self.data_dic
#
#     # TODO: �����ȼ� ʵ���Զ���lag
#     #  �����ȼ� ģ���������� �Ƚϸ��� ������ݷ���ʱ���Ż���û����֮ǰͳһ��֮ǰ�����ݣ����⺯��Ӧ���������Ը��õ��ܶ�����
#     def series_to_supervised(self, n_in=1, n_out=1, dropnan=True):
#         # Ĭ��Y���䣬X��δ����һ���²��������൱�����ϸ��µ�X��Ԥ������µ�Y������ģ������²ŵõ��ϸ��µĺ�����ݡ�
#         # �����Ҫ����ʱ��feature�ɵ���n_in��ͬ��n_out��
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
#             # ֻ����x y���е�����
#             common_ind = data_dic['x'].index.intersection(data_dic['y'].index)
#             data_dic['x'] = data_dic['x'].loc[common_ind]
#             data_dic['y'] = data_dic['y'].loc[common_ind]
#
#         return data_dic
#
#     # TODO: �����ȼ� �����������������ϸ�Ż����ݻ�ȡʱ����Ҫ���������֡�
#     # def fill_x_na(self):
#     #     # ���Ի�������Ƶdf��ע��������ʼͳ��������ݲ�û�н���bfill
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
#         # ��fillna��resample
#         def get_y():
#             self.s_datay.fillna(method='ffill', inplace=True)
#             # ����һЩ����������ʼʱ�����Ʒ����Ҫ����
#             self.s_datay.fillna(method='bfill', inplace=True)
#             df = deepcopy(self.s_datay)
#             # ��ȡ�����д��ڵ���ĩ���ڣ����ݸ�����year��month����groupby,Ȼ��ȡÿ�����������max�������ڣ�����ŵ�month_end�б��С�
#             # ��Щ���࣬�����뱣����ĩ���ڣ��������ֺ�x��ĩ���ڶԲ��ϣ���ֻ�����¡�
#             df['date'] = df.index
#             df['year'] = df['date'].apply(lambda x: x.year)
#             df['month'] = df['date'].apply(lambda x: x.month)
#             grouped = df['date'].groupby([df['year'], df['month']])
#             month_end = grouped.max().to_list()
#             df.drop(['year', 'month', 'date'], axis=1, inplace=True)
#             # ת���������ʣ��ʼ��һ����ĩû�������ʣ����һ���²�������������
#             month_end_df = df.loc[month_end, :].pct_change()
#             # index����ĩ�����ն��뵽�·�
#             month_end_df.index = month_end_df.index.to_period('M')
#             return month_end_df
#
#         # ���x fillna��resample
#         def get_x():
#             # �����ʲ�����
#             df = pd.concat([self.s_datax, self.s_datay.copy()], axis=1)
#             df = fill_x_na(df, self.info)
#             # ����groupby��ֱ��resample
#             month_end_df = pd.DataFrame()
#             for id in df.columns:
#                 if self.info.loc[id, 'resample(��)'] == 'last':
#                     ts = df.loc[:, id].resample('1M').last()
#                 elif self.info.loc[id, 'resample(��)'] == 'avg':
#                     ts = df.loc[:, id].resample('1M').mean()
#                 elif self.info.loc[id, 'resample(��)'] == 'cumsum':
#                     # ��עһ�����ݱ�transform��
#                     ts = df.loc[:, id].cumsum().resample('1M').last()
#                     self.info.loc[id, 'ָ������'] = 'cumsumed' + self.info.loc[id, 'ָ������']
#                 else:  # ʣ�µ�Ӧ�����ʲ����ƣ�ֱ��ȡ��ĩ
#                     ts = df.loc[:, id].resample('1M').last()
#                 month_end_df = pd.concat([month_end_df, ts], axis=1)
#             # index����ĩ�����ն��뵽�·�
#             # noinspection PyTypeChecker
#             month_end_df.index = pd.to_datetime(month_end_df.index).to_period('M')
#             return fill_na_naive(month_end_df)
#
#         # Y��ʼĩ�²����ã�X����
#         y = get_y()[1: -1]
#         x = get_x()[1: -1]
#         # �����ʲ�����
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
#         # MLF���� M5528820(�ɡ�ɾ������)ƴ�ӵ�M0329545(��)
#         if 'M5528820' in self.s_datax.columns:
#             self.s_datax.loc[:, 'M0329545'] = self.s_datax.M5528820.copy().add(self.s_datax.M0329545.copy(),
#                                                                                fill_value=0).copy()
#             self.s_datax.drop('M5528820', axis=1, inplace=True)
#         # �ۼ�ֵת��Ϊ����ֵ
#         for id in self.s_datax.columns:
#             if self.info.loc[id, '�Ƿ��ۼ�ֵ'] == '��':
#                 self.info.loc[id, 'ָ������'] = '(�¶Ȼ�)' + self.info.loc[id, 'ָ������']
#                 sr = self.s_datax[id]
#                 sr_ori = deepcopy(sr)
#                 for date in sr.index:
#                     if np.isfinite(sr[date]):
#                         try:
#                             diff = sr[date] - sr_ori[date - pd.offsets.MonthEnd()]
#                             self.s_datax.loc[date, id] = diff if diff > 0 else sr[date]
#                         except:
#                             pass
#         # ȥ��һ��Ԥ�����ݣ��Ƚ�������δ�����ݰ�
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
#     # ���ֳ��ܵ�ѵ��������ѵ���Ӻ���֤��tscv���Ͳ��Լ���
#     def get_split_data(self, train_pct):
#         if len(self.data_dic['x']) != len(self.data_dic['y']):
#             raise Exception('������x y���в��ȳ�')
#         length = int(len(self.data_dic['x']) * train_pct)
#         train_x = self.data_dic['x'].iloc[0:length]
#         train_y = self.data_dic['y'].iloc[0:length]
#         test_x = self.data_dic['x'].iloc[length:]
#         test_y = self.data_dic['y'].iloc[length:]
#         return {'x': train_x, 'y': train_y}, {'x': test_x, 'y': test_y}
#
#     # in-sample
#     def train(self):
#         # TimeSeriesSplit�е�gap�Ǹ���ģ�Ӧ���Ǹ�ͳ��ѧԭ���йء���֪�Ƿ���Ҫ��
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
#         # TODO: MultiOutput��ֱ��regress���output�к�����
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
