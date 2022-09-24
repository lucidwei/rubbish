# coding=gbk
# lucid   当前系统用户
# 2022/8/16   当前系统日期
# 14:47   当前系统时间
# PyCharm   创建文件的IDE名称
import pickle, os.path, platform
from os.path import abspath
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
import pipe_preproc
import utils_eda


# 跨平台文件路径处理
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


# 针对wind excel数据复用
# minor warning: wind下载的自定义合成指标没有完整的指标ID，列名有极小的概率重叠，其或造成feature丢失。8月19日版本数据暂无此问题。
class GetStructuralData:
    # 对象存储数据（字典类型）和描述表，类方法功能包括1格式化（取值变更坐标）2将数据和描述表存入data文件夹
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
        if not os.path.exists(get_path('info_table')) or self.update:
            info.to_csv(get_path('info_table'))
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
        # 获取结构化数据实例，update为True则读取wind源数据，默认False读取之前保存的结构化数据文件
        s_data = GetStructuralData(ori_data_path, update=if_update)
        raw_x = s_data.structural_data['x'][begT:endT].copy()
        raw_y = s_data.structural_data['y'].iloc[::-1][begT:endT].copy()

        # 涉及y变换的sk不支持，要拆开
        info = pd.read_excel(info_path, index_col=0, engine="openpyxl")
        pipe_preprocess0 = Pipeline(steps=[
            ('special_treatment', pipe_preproc.SpecialTreatment(info)),
            ('data_alignment', pipe_preproc.DataAlignment(align_to, info)),
        ])
        # pipe_preprocess1 = Pipeline(steps=[
        #     # 处理稳态不能用bfill过的X
        #     ('station_origin', pipe_preproc.GetStationary()),
        #     # 丢失坐标
        #     ('imputer', KNNImputer()),
        #     ('ts_to_supervised', pipe_preproc.SeriesToSupervised(n_in=use_x_lags))
        # ])

        X0, y0 = pipe_preprocess0.fit_transform(raw_x, raw_y)
        X1 = pipe_preproc.GetStationary().transform(X0)
        X2 = KNNImputer().fit_transform(X1)
        X3 = pd.DataFrame(X2, index=X1.index, columns=X1.columns)
        X, y = pipe_preproc.SeriesToSupervised(n_in=use_x_lags).transform(X3, y0)

        # selectFromModel中y不能有空
        # TODO: 草率处理，严谨应该在data_alignment中完善逻辑
        y_filled = y.fillna(method='ffill').fillna(method='bfill')
        print('...Pre-processing finished\n')

        # 缓存数据
        with open(cache_path, 'wb') as f:
            pickle.dump((X, y_filled), f)
        print('data pickle saved')
    else:
        # 读取缓存数据
        with open(cache_path, 'rb') as f:
            (X, y) = pickle.load(f)
        print('data pickle loaded')

    return X, y


def reg_to_class(y, tile_num):
    df = pd.DataFrame(index=y.index, columns=y.columns)
    if tile_num == 3:
        for col_ind, col in y.iteritems():
            df.loc[:, col_ind] = pd.qcut(col, 3, labels=False) #, labels=['低配', '持平', '超配'])
    else:
        raise Exception('其他分类暂不支持')
    return df


def add_2years_test(X_train, X_test):
    # 先默认月度数据
    X_test = pd.concat([X_train.iloc[-26:, :], X_test])
    # y_test = pd.concat([y_train.iloc[-24:, :], y_test])
    return X_test


from sklearn.feature_selection import SelectFromModel, SelectKBest, mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV


# 想要得到某个资产选择出的features原始id，每个原始id在10个资产中被选出来的次数
# 针对预处理生成的X进行筛选
# 返回多个特征筛选方法选择出来的原始数据ID的并集
def single_y_selector(X, yi, i):
    num_select = 50
    lr = SelectFromModel(estimator=LinearRegression(), max_features=num_select).fit(X, yi)
    ridge = SelectFromModel(estimator=RidgeCV(), max_features=num_select).fit(X, yi)
    sgd = SelectFromModel(estimator=SGDRegressor(), max_features=num_select).fit(X, yi)
    rf = SelectFromModel(estimator=RandomForestRegressor(random_state=1996), max_features=num_select).fit(X, yi)
    gbr = SelectFromModel(estimator=GradientBoostingRegressor(random_state=1996), max_features=num_select).fit(X, yi)
    # MI太慢了，占整个时间的一半，而且选出来的好像比较奇怪
    mi = SelectKBest(score_func=mutual_info_regression, k=num_select).fit(X, yi)
    common_support = lr.get_support() | ridge.get_support() | sgd.get_support() | rf.get_support() | \
                     gbr.get_support() | mi.get_support()
    print('%d features selected for asset %d' % (sum(common_support), i))
    selected = X.columns[common_support]
    selected_ori = np.unique(utils_eda.trans_series_id(selected))
    print('%d original features selected for asset %d' % (len(selected_ori), i))
    return selected_ori


# 多个资产筛选出来的原始ID取并集
def manual_features_selector(X, y):
    final = pd.Series()
    l = len(y.columns)
    for i in range(l):
        final = np.union1d(final, single_y_selector(X, y.iloc[:, i], i))
    print('%d original features selected for all assets' % (len(final)))
    return final


def get_garbage_features(selected_id):
    info = utils_eda.get_info()
    # info-selected取差集
    garbage = list(set(info.index).difference(set(selected_id)))
    names = pd.Series(garbage).apply(lambda x: utils_eda.get_ori_name(x, info))
    # id和名称保存到data_dump
    garbage_info = pd.DataFrame({'ID': garbage, '名称': names})
    garbage_info.to_csv(r'.\data\garbage.csv')
    return garbage


# TODO: 中优先级 帮助好像不大，放到preproc里
def remove_garbage_data(X, selected_id):
    pass
