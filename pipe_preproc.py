# coding=gbk
# lucid   当前系统用户
# 2022/9/1   当前系统日期
# 17:02   当前系统时间
# PyCharm   创建文件的IDE名称

from copy import deepcopy
import warnings
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss

import utils_eda
from utils_eda import get_ori_id


class SpecialTreatment(BaseEstimator):
    def __init__(self, info: pd.DataFrame):
        self.info = info
        print('...initializing SpecialTreatment\n')

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        print('...transforming SpecialTreatment \n')
        # MLF利率 M5528820(旧、删除该列)拼接到M0329545(新)
        if 'M5528820' in X.columns:
            X.loc[:, 'M0329545'] = X.M5528820.add(X.M0329545, fill_value=0).copy()
            X.drop('M5528820', axis=1, inplace=True)
        # 累计值转化为当月值
        for id in X.columns:
            if self.info.loc[id, '是否累计值'] == '是':
                self.info.loc[id, '指标名称'] = '(月度化)' + self.info.loc[id, '指标名称']
                sr = X[id]
                sr_ori = deepcopy(sr)
                for date in sr.index:
                    if np.isfinite(sr[date]):
                        try:
                            diff = sr[date] - sr_ori[date - pd.offsets.MonthEnd()]
                            X.loc[date, id] = diff if diff > 0 else sr[date]
                        except:
                            pass
        # 关于可疑的一致预期数据，先仅仅掐掉未来数据，先不用在这里处理
        return X, y

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)


class DataAlignment(BaseEstimator):
    def __init__(self, align_to, info):
        self.align_to = align_to
        self.info = info
        print('...initializing DataAlignment\n')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('...transforming DataAlignment \n')
        ###### unpack X if is tuple X = (X,y), otherwise pass y through
        if isinstance(X, tuple):
            X, y = X
        if self.align_to == 'month':
            return self.align_to_month(X, y)
        elif self.align_to == 'week':
            return self.align_to_week(X, y)
        else:
            raise Exception('Need to specify alignment frequency')

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)

    def align_to_month(self, X, y):
        # 先fillna再resample
        def get_y():
            y.fillna(method='ffill', inplace=True)
            # 对于一些交易数据起始时间晚的品种需要补空.先不着急bfill
            # y.fillna(method='bfill', inplace=True)
            df = deepcopy(y)
            # 获取数据中存在的月末日期：根据辅助列year和month进行groupby,然后取每个分组中最大（max）的日期，并存放到month_end列表中。
            # 有些冗余，当初想保留月末日期，后来发现和x月末日期对不上，故只保留月。
            df['date'] = df.index
            df['year'] = df['date'].apply(lambda x: x.year)
            df['month'] = df['date'].apply(lambda x: x.month)
            grouped = df['date'].groupby([df['year'], df['month']])
            month_end = grouped.max().to_list()
            df.drop(['year', 'month', 'date'], axis=1, inplace=True)
            # 转换成收益率，最开始第一个月末没有收益率，最后一个月不是完整月收益
            month_end_df = df.loc[month_end, :].pct_change()
            # index从月末交易日对齐到月份
            month_end_df.index = month_end_df.index.to_period('M')
            return month_end_df

        # 针对x fillna和resample
        def get_x():
            # 插入资产走势
            df = pd.concat([X, y.copy()], axis=1)
            df = fill_x_na(df, self.info)
            # 可以groupby后直接resample
            month_end_df = pd.DataFrame()
            for id in df.columns:
                if self.info.loc[id, 'resample(月)'] == 'last':
                    ts = df.loc[:, id].resample('1M').last()
                elif self.info.loc[id, 'resample(月)'] == 'avg':
                    ts = df.loc[:, id].resample('1M').mean()
                elif self.info.loc[id, 'resample(月)'] == 'cumsum':
                    # 备注一下数据被transform了
                    ts = df.loc[:, id].cumsum().resample('1M').last()
                    self.info.loc[id, '指标名称'] = 'cumsumed' + self.info.loc[id, '指标名称']
                else:  # 剩下的应该是资产走势，直接取月末
                    ts = df.loc[:, id].resample('1M').last()
                month_end_df = pd.concat([month_end_df, ts], axis=1)
            # index从月末交易日对齐到月份
            # noinspection PyTypeChecker
            month_end_df.index = pd.to_datetime(month_end_df.index).to_period('M')

            return month_end_df.fillna(method='ffill')

        # Y的始末月不能用，X对齐
        y_return = get_y()[1: -1]
        x = get_x()[1: -1]
        # 插入资产收益
        y_return.columns = y_return.columns.map(lambda x: x + '_returns')
        x = pd.concat([x, y_return], axis=1)
        return x, y_return

    # TODO: 低优先级 周度align规则尚需手动完成
    def align_to_week(self, X, y=None):
        X = X.resample('1w').ffill()
        y = y.resample('1w').ffill()
        return X, y


def fill_na_naive(df):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df


# TODO: 低优先级 单拎出来，后续若详细优化数据获取时间需要在这里下手。
def fill_x_na(df, info_df):
    # 个性化操作高频df，注：对于起始统计晚的数据并没有进行bfill
    for id in df.columns:
        if info_df.loc[id, 'fillna'] == 'ffill':
            df.loc[:, id].fillna(method='ffill', inplace=True)
        elif info_df.loc[id, 'fillna'] == '0fill':
            df.loc[:, id].fillna(value=0, inplace=True)
        elif pd.isnull(info_df.loc[id, 'fillna']):
            pass
        else:
            raise Exception('donno how to fillna')
    return df


class GetStationary(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.info = utils_eda.get_info()
        print('...initializing GetStationary\n')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, tuple):
            X, y = X
        print('...transforming GetStationary \n')
        # TODO: 低优先级 有些规律性空值的或许有计算差错（未验证猜想）
        df = X.fillna(method='ffill')
        record = {col_ind: station_test(col) for col_ind, col in df.iteritems()}
        count = pd.value_counts(list(record.values()))
        # print('There are %d not stationary variables' % (len(record) - count['stationary']))
        # 不确定trend有没有用，trend和residual都留着吧
        for col_ind, col in df.iteritems():
            # 非稳态且单位非'%'才进行操作
            ori_id = get_ori_id(col_ind)
            if record[col_ind] != 'stationary' and self.info.loc[ori_id, '单位'] != '%':
                stl = STL(col.dropna(), period=12, robust=True)
                decomposed = stl.fit()
                df.insert(df.columns.get_loc(col_ind)+1, column=col_ind + '_trend', value=decomposed.trend)
                # TODO: 低优先级 可能需要处理个别数据的outliers
                df.insert(df.columns.get_loc(col_ind)+2, column=col_ind + '_resid', value=decomposed.resid)
                df = df.copy()
                # 按理说应该drop，但万一原始数据也有用呢
                # df.drop(col_ind, inplace=True, axis=1)
        df.fillna(method='bfill', inplace=True)
        # y.fillna(method='bfill', inplace=True)
        return df


def station_test(ts):
    def kpss_test(timeseries):
        # print("Results of KPSS Test:")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpsstest = kpss(timeseries, regression="c", nlags="auto")
        kpss_output = pd.Series(
            kpsstest[0:3], index=["Test Statistic", "p-value", "Lags Used"]
        )
        # print(kpss_output)
        return kpss_output[1]

    def adf_test(timeseries):
        # print("Results of Dickey-Fuller Test:")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dftest = adfuller(timeseries, autolag="AIC")
        dfoutput = pd.Series(
            dftest[0:4],
            index=[
                "Test Statistic",
                "p-value",
                "#Lags Used",
                "Number of Observations Used",
            ], )
        # print(dfoutput)
        return dfoutput[1]

    ts_new = ts.dropna()
    p_adf = adf_test(ts_new)
    p_kpss = kpss_test(ts_new)
    threshold = 0.05
    if p_adf <= threshold <= p_kpss:
        return 'stationary'
    elif p_kpss < threshold < p_adf:
        return 'non-stationary'
    elif p_adf > threshold and p_kpss > threshold:
        return 'trend-stationary'
    elif p_adf < threshold and p_kpss < threshold:
        return 'diff-stationary'
    else:
        raise Exception('donno stationarity')


# TODO: 中优先级 实现自定义lag
#  低优先级 模拟生产环境 比较复杂 针对数据发布时间优化，没发布之前统一用之前的数据，另外函数应该做到可以复用到周度数据
class SeriesToSupervised(BaseEstimator):
    def __init__(self, n_in=1, n_out=1, dropnan=True):
        print('...initializing SeriesToSupervised\n')
        self.n_in = n_in
        self.n_out = n_out
        self.dropnan = dropnan

    def fit(self, X, y):
        return self

    def transform(self, X, y):
        if isinstance(X, tuple):
            X, y = X
        print('...transforming SeriesToSupervised\n')
        # 默认Y不变，X向未来移一个月并保留，相当于用上个月的X来预测这个月的Y。可以模拟这个月才得到上个月的宏观数据。
        # 如果需要增加时移feature可调大n_in，同理n_out。
        dfs, col_names = list(), list()
        # input sequence (t-n, ... t-1)
        x_df = X.copy()
        for i in range(self.n_in, 0, -1):
            dfs.append(x_df.shift(i))
            col_names += [('var%d(t-%d)' % (j + 1, i) + x_df.columns.values[j]) for j in range(x_df.shape[1])]
        X = pd.concat(dfs, axis=1)
        X.columns = col_names

        # forecast sequence (t, t+1, ... t+n)
        dfs, col_names = list(), list()
        y_df = y.copy()
        for i in range(0, self.n_out):
            dfs.append(y_df.shift(-i))
            if i == 0:
                col_names += [('var%d(t)' % (j + 1) + y_df.columns.values[j]) for j in range(y_df.shape[1])]
            else:
                col_names += [('var%d(t+%d)' % (j + 1, i) + y_df.columns.values[j]) for j in range(y_df.shape[1])]
        y = pd.concat(dfs, axis=1)
        y.columns = col_names

        if self.dropnan:
            # drop rows with NaN values
            X.dropna(inplace=True)
            y.dropna(inplace=True)
            # 只保留x y共有的日期
            common_ind = X.index.intersection(y.index)
            X = X.loc[common_ind]
            y = y.loc[common_ind]

        return X, y

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
