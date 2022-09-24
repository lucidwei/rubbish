# coding=gbk
# lucid   ��ǰϵͳ�û�
# 2022/8/16   ��ǰϵͳ����
# 14:47   ��ǰϵͳʱ��
# PyCharm   �����ļ���IDE����
import pickle, os.path, platform
from os.path import abspath
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import pipe_preproc
import utils_eda


# ��ƽ̨�ļ�·������
def get_path(file_name: str):
    windows_dic = {
        'data_x': r'.\data\data_x.csv',
        'data_y': r'.\data\data_y.csv',
        'to_month_table': r'.\data\to_month_table.xlsx',
        'to_week_table': r'.\data\to_week_table.xlsx',
        'info_table': r'.\data\info_table.csv',
        'prepipe_data': r'data_dump\prepipe_data',
        'cachedir': 'C:\\Downloads\\tpot_cache'
    }

    mac_dic = {
        'data_x': './data/data_x.csv',
        'data_y': './data/data_y.csv',
        'to_month_table': './data/to_month_table.xlsx',
        'to_week_table': './data/to_week_table.xlsx',
        'info_table': './data/info_table.csv',
        'prepipe_data': './data_dump/prepipe_data',
        'cachedir': abspath('../../Documents/tpot_cache')
    }

    return windows_dic[file_name] if platform.system().lower() == 'windows' else mac_dic[file_name]


# ���wind excel���ݸ���
# minor warning: wind���ص��Զ���ϳ�ָ��û��������ָ��ID�������м�С�ĸ����ص���������feature��ʧ��8��19�հ汾�������޴����⡣
class GetStructuralData:
    # ����洢���ݣ��ֵ����ͣ����������෽�����ܰ���1��ʽ����ȡֵ������꣩2�����ݺ����������data�ļ���
    def __init__(self, path_read, update=False):
        print('update:', update)
        print('os.path.exists(x&y csv):', os.path.exists(get_path('data_x') and os.path.exists(get_path('data_y'))))
        self.path_read = path_read
        self.update = update
        if update and os.path.exists(get_path('data_x')):
            self.raw_data = self.read_files()
            self.info = self.get_info(last_info_row=6)
            self.structural_data = self.get_sdata(last_info_row=6)
        else:
            self.info = pd.read_csv(get_path('info_table'), index_col=0, parse_dates=True)
            self.structural_data = {'x': pd.read_csv(get_path('data_x'), index_col=0, parse_dates=True),
                                    'y': pd.read_csv(get_path('data_y'), index_col=0, parse_dates=True)}

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
        if not os.path.exists(get_path('info_table')) or self.update:
            info.to_csv(get_path('info_table'))
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
        if not os.path.exists(get_path('data_x')) or self.update:
            sdata_x.to_csv(get_path('data_x'))
            sdata_y.to_csv(get_path('data_y'))
        return {'x': sdata_x, 'y': sdata_y}


def get_preproc_data(ori_data_path, if_update, use_cache, use_x_lags, align_to, begT, endT):
    cache_path = get_path('prepipe_data')
    if align_to == 'month':
        info_path = get_path('to_month_table')
    elif align_to == 'week':
        info_path = get_path('to_week_table')
    else:
        raise Exception('Please specify alignment frequency')

    if not use_cache:
        # ��ȡ�ṹ������ʵ����updateΪTrue���ȡwindԴ���ݣ�Ĭ��False��ȡ֮ǰ����Ľṹ�������ļ�
        s_data = GetStructuralData(ori_data_path, update=if_update)
        raw_x = s_data.structural_data['x'][begT:endT].copy()
        raw_y = s_data.structural_data['y'].iloc[::-1][begT:endT].copy()

        # �漰y�任��sk��֧�֣�Ҫ��
        info = pd.read_excel(info_path, index_col=0, engine="openpyxl")
        pipe_preprocess0 = Pipeline(steps=[
            ('special_treatment', pipe_preproc.SpecialTreatment(info)),
            ('data_alignment', pipe_preproc.DataAlignment(align_to, info)),
        ])
        # pipe_preprocess1 = Pipeline(steps=[
        #     # ������̬������bfill����X
        #     ('station_origin', pipe_preproc.GetStationary()),
        #     # ��ʧ����
        #     ('imputer', KNNImputer()),
        #     ('ts_to_supervised', pipe_preproc.SeriesToSupervised(n_in=use_x_lags))
        # ])

        X0, y0 = pipe_preprocess0.fit_transform(raw_x, raw_y)
        X1 = pipe_preproc.GetStationary().transform(X0)
        X2 = KNNImputer().fit_transform(X1)
        X3 = pd.DataFrame(X2, index=X1.index, columns=X1.columns)
        X, y = pipe_preproc.SeriesToSupervised(n_in=use_x_lags).transform(X3, y0)

        # selectFromModel��y�����п�
        # TODO: ���ʴ����Ͻ�Ӧ����data_alignment�������߼�
        y_filled = y.fillna(method='ffill').fillna(method='bfill')
        print('...Pre-processing finished\n')

        # ��������
        with open(cache_path, 'wb') as f:
            pickle.dump((X, y_filled), f)
        print('data pickle saved')
    else:
        # ��ȡ��������
        with open(cache_path, 'rb') as f:
            (X, y) = pickle.load(f)
        print('data pickle loaded')

    return X, y


def reg_to_class(y, tile_num):
    df = pd.DataFrame(index=y.index, columns=y.columns)
    if tile_num == 3:
        for col_ind, col in y.iteritems():
            df.loc[:, col_ind] = pd.qcut(col, 3, labels=False) #, labels=['����', '��ƽ', '����'])
    else:
        raise Exception('���������ݲ�֧��')
    return df


def add_2years_test(X_train, X_test):
    # ��Ĭ���¶�����
    X_test = pd.concat([X_train.iloc[-26:, :], X_test])
    # y_test = pd.concat([y_train.iloc[-24:, :], y_test])
    return X_test


from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV


# ��Ҫ�õ�ĳ���ʲ�ѡ�����featuresԭʼid��ÿ��ԭʼid��10���ʲ��б�ѡ�����Ĵ���
# ���Ԥ�������ɵ�X����ɸѡ
# ���ض������ɸѡ����ѡ�������ԭʼ����ID�Ĳ���
def single_y_selector(X, yi, i):
    num_select = 50
    lr = SelectFromModel(estimator=LinearRegression(), max_features=num_select).fit(X, yi)
    ridge = SelectFromModel(estimator=RidgeCV(), max_features=num_select).fit(X, yi)
    sgd = SelectFromModel(estimator=SGDRegressor(), max_features=num_select).fit(X, yi)
    rf = SelectFromModel(estimator=RandomForestRegressor(random_state=1996), max_features=num_select).fit(X, yi)
    gbr = SelectFromModel(estimator=GradientBoostingRegressor(random_state=1996), max_features=num_select).fit(X, yi)
    # MI̫���ˣ�ռ����ʱ���һ�룬����ѡ�����ĺ���Ƚ����
    mi = SelectKBest(score_func=mutual_info_regression, k=num_select).fit(X, yi)
    common_support = lr.get_support() | ridge.get_support() | sgd.get_support() | rf.get_support() | \
                     gbr.get_support() | mi.get_support()
    print('%d features selected for asset %d' % (sum(common_support), i))
    selected = X.columns[common_support]
    selected_ori = np.unique(utils_eda.trans_series_id(selected))
    print('%d original features selected for asset %d' % (len(selected_ori), i))
    return selected_ori


# ����ʲ�ɸѡ������ԭʼIDȡ����
def manual_features_selector(X, y):
    final = pd.Series()
    l = len(y.columns)
    for i in range(l):
        final = np.union1d(final, single_y_selector(X, y.iloc[:, i], i))
    print('%d original features selected for all assets' % (len(final)))
    return final


def get_garbage_features(selected_id):
    info = utils_eda.get_info()
    # info-selectedȡ�
    garbage = list(set(info.index).difference(set(selected_id)))
    names = pd.Series(garbage).apply(lambda x: utils_eda.get_ori_name(x, info))
    # id�����Ʊ��浽data_dump
    garbage_info = pd.DataFrame({'ID': garbage, '����': names})
    garbage_info.to_csv(r'.\data\garbage.csv')
    return garbage


# TODO: �����ȼ� �������񲻴󣬷ŵ�preproc��
def remove_garbage_data(X, selected_id):
    pass
