# coding=gbk
# lucid   ��ǰϵͳ�û�
# 2022/9/1   ��ǰϵͳ����
# 17:02   ��ǰϵͳʱ��
# PyCharm   �����ļ���IDE����

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
        # MLF���� M5528820(�ɡ�ɾ������)ƴ�ӵ�M0329545(��)
        if 'M5528820' in X.columns:
            X.loc[:, 'M0329545'] = X.M5528820.add(X.M0329545, fill_value=0).copy()
            X.drop('M5528820', axis=1, inplace=True)
        # �ۼ�ֵת��Ϊ����ֵ
        for id in X.columns:
            if self.info.loc[id, '�Ƿ��ۼ�ֵ'] == '��':
                self.info.loc[id, 'ָ������'] = '(�¶Ȼ�)' + self.info.loc[id, 'ָ������']
                sr = X[id]
                sr_ori = deepcopy(sr)
                for date in sr.index:
                    if np.isfinite(sr[date]):
                        try:
                            diff = sr[date] - sr_ori[date - pd.offsets.MonthEnd()]
                            X.loc[date, id] = diff if diff > 0 else sr[date]
                        except:
                            pass
        # ���ڿ��ɵ�һ��Ԥ�����ݣ��Ƚ�������δ�����ݣ��Ȳ��������ﴦ��
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
        # ��fillna��resample
        def get_y():
            y.fillna(method='ffill', inplace=True)
            # ����һЩ����������ʼʱ�����Ʒ����Ҫ����.�Ȳ��ż�bfill
            # y.fillna(method='bfill', inplace=True)
            df = deepcopy(y)
            # ��ȡ�����д��ڵ���ĩ���ڣ����ݸ�����year��month����groupby,Ȼ��ȡÿ�����������max�������ڣ�����ŵ�month_end�б��С�
            # ��Щ���࣬�����뱣����ĩ���ڣ��������ֺ�x��ĩ���ڶԲ��ϣ���ֻ�����¡�
            df['date'] = df.index
            df['year'] = df['date'].apply(lambda x: x.year)
            df['month'] = df['date'].apply(lambda x: x.month)
            grouped = df['date'].groupby([df['year'], df['month']])
            month_end = grouped.max().to_list()
            df.drop(['year', 'month', 'date'], axis=1, inplace=True)
            # ת���������ʣ��ʼ��һ����ĩû�������ʣ����һ���²�������������
            month_end_df = df.loc[month_end, :].pct_change()
            # index����ĩ�����ն��뵽�·�
            month_end_df.index = month_end_df.index.to_period('M')
            return month_end_df

        # ���x fillna��resample
        def get_x():
            # �����ʲ�����
            df = pd.concat([X, y.copy()], axis=1)
            df = fill_x_na(df, self.info)
            # ����groupby��ֱ��resample
            month_end_df = pd.DataFrame()
            for id in df.columns:
                if self.info.loc[id, 'resample(��)'] == 'last':
                    ts = df.loc[:, id].resample('1M').last()
                elif self.info.loc[id, 'resample(��)'] == 'avg':
                    ts = df.loc[:, id].resample('1M').mean()
                elif self.info.loc[id, 'resample(��)'] == 'cumsum':
                    # ��עһ�����ݱ�transform��
                    ts = df.loc[:, id].cumsum().resample('1M').last()
                    self.info.loc[id, 'ָ������'] = 'cumsumed' + self.info.loc[id, 'ָ������']
                else:  # ʣ�µ�Ӧ�����ʲ����ƣ�ֱ��ȡ��ĩ
                    ts = df.loc[:, id].resample('1M').last()
                month_end_df = pd.concat([month_end_df, ts], axis=1)
            # index����ĩ�����ն��뵽�·�
            # noinspection PyTypeChecker
            month_end_df.index = pd.to_datetime(month_end_df.index).to_period('M')

            return month_end_df.fillna(method='ffill')

        # Y��ʼĩ�²����ã�X����
        y_return = get_y()[1: -1]
        x = get_x()[1: -1]
        # �����ʲ�����
        y_return.columns = y_return.columns.map(lambda x: x + '_returns')
        x = pd.concat([x, y_return], axis=1)
        return x, y_return

    # TODO: �����ȼ� �ܶ�align���������ֶ����
    def align_to_week(self, X, y=None):
        X = X.resample('1w').ffill()
        y = y.resample('1w').ffill()
        return X, y


def fill_na_naive(df):
    df.fillna(method='ffill', inplace=True)
    df.fillna(method='bfill', inplace=True)
    return df


# TODO: �����ȼ� �����������������ϸ�Ż����ݻ�ȡʱ����Ҫ���������֡�
def fill_x_na(df, info_df):
    # ���Ի�������Ƶdf��ע��������ʼͳ��������ݲ�û�н���bfill
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
        # TODO: �����ȼ� ��Щ�����Կ�ֵ�Ļ����м�����δ��֤���룩
        df = X.fillna(method='ffill')
        record = {col_ind: station_test(col) for col_ind, col in df.iteritems()}
        count = pd.value_counts(list(record.values()))
        # print('There are %d not stationary variables' % (len(record) - count['stationary']))
        # ��ȷ��trend��û���ã�trend��residual�����Ű�
        for col_ind, col in df.iteritems():
            # ����̬�ҵ�λ��'%'�Ž��в���
            ori_id = get_ori_id(col_ind)
            if record[col_ind] != 'stationary' and self.info.loc[ori_id, '��λ'] != '%':
                stl = STL(col.dropna(), period=12, robust=True)
                decomposed = stl.fit()
                df.insert(df.columns.get_loc(col_ind)+1, column=col_ind + '_trend', value=decomposed.trend)
                # TODO: �����ȼ� ������Ҫ����������ݵ�outliers
                df.insert(df.columns.get_loc(col_ind)+2, column=col_ind + '_resid', value=decomposed.resid)
                df = df.copy()
                # ����˵Ӧ��drop������һԭʼ����Ҳ������
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


# TODO: �����ȼ� ʵ���Զ���lag
#  �����ȼ� ģ���������� �Ƚϸ��� ������ݷ���ʱ���Ż���û����֮ǰͳһ��֮ǰ�����ݣ����⺯��Ӧ���������Ը��õ��ܶ�����
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
        # Ĭ��Y���䣬X��δ����һ���²��������൱�����ϸ��µ�X��Ԥ������µ�Y������ģ������²ŵõ��ϸ��µĺ�����ݡ�
        # �����Ҫ����ʱ��feature�ɵ���n_in��ͬ��n_out��
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
            # ֻ����x y���е�����
            common_ind = X.index.intersection(y.index)
            X = X.loc[common_ind]
            y = y.loc[common_ind]

        return X, y

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X, y)
