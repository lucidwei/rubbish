# coding=gbk
# lucid   当前系统用户
# 2022/8/16   当前系统日期
# 14:47   当前系统时间
# PyCharm   创建文件的IDE名称
import pickle, os.path, platform
from os.path import abspath
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from tpot import TPOTRegressor
from joblib import Memory
import pipe_preproc
from pipe_models import *


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


def get_preproc_data(ori_data_path, if_update, use_cache, align_to, use_lag_x, begT, endT):
    if not use_cache:
        # 获取结构化数据实例，update为True则读取wind源数据，默认False读取之前保存的结构化数据文件
        s_data = GetStructuralData(ori_data_path, update=if_update)
        raw_x = s_data.structural_data['x'][begT:endT].copy()
        raw_y = s_data.structural_data['y'].iloc[::-1][begT:endT].copy()

        # 预处理
        info = pd.read_excel(r'.\data\to_month_table.xlsx', index_col=0,
                             engine="openpyxl") if align_to == 'month' else \
            pd.read_excel(r'.\data\to_week_table.xlsx', index_col=0, engine="openpyxl")
        pipe_preprocess = Pipeline(steps=[
            ('special_treatment', pipe_preproc.SpecialTreatment(info)),
            ('data_alignment', pipe_preproc.DataAlignment(align_to, info)),
            ('get_stationary', pipe_preproc.GetStationary(info)),
            ('series_to_supervised', pipe_preproc.SeriesToSupervised(n_in=use_lag_x, n_out=1))
        ])

        X, y = pipe_preprocess.fit_transform(raw_x, raw_y)
        print('...Pre-processing finished\n')

        # 缓存数据
        with open('debug/prepipe_data', 'wb') as f:
            pickle.dump((X, y), f)
        print('data pickle saved')
    else:
        # 读取缓存数据
        with open('debug/prepipe_data', 'rb') as f:
            (X, y) = pickle.load(f)
        print('data pickle loaded')

    return X, y


def generate_1_pipe(X, y, generations, population_size, max_time_mins, cachedir, tpot_config=None, pipe_num=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8, test_size=0.2,
                                                        shuffle=False)
    memory = Memory(location=cachedir, verbose=0)
    cv = TimeSeriesSplit()
    pipeline_optimizer = TPOTRegressor(generations=generations, population_size=population_size, cv=cv,
                                       template='Selector-Transformer-Regressor',
                                       scoring='r2',
                                       early_stop=4,
                                       config_dict=tpot_config,
                                       max_time_mins=max_time_mins,
                                       memory=memory,
                                       warm_start=True,
                                       periodic_checkpoint_folder=abspath('../../Documents/tpot_checkpoint'),
                                       log_file=abspath('../../Documents/tpot_log/log'+str(pipe_num)),
                                       random_state=1996, verbosity=3)
    pipeline_optimizer.fit(X_train, y_train)
    print('A pipe finised, score(X_test, y_test):', pipeline_optimizer.score(X_test, y_test))
    if pipe_num is None:
        pipeline_optimizer.export('./tpot_gen/multioutput_tpotpipe.py')
    else:
        pipeline_optimizer.export('./tpot_gen/separate_tpotpipe%d.py' % pipe_num)

    return pipeline_optimizer, X_test, y_test


tpot_config = {
    'sklearn.feature_selection.SelectFromModel': {
        # 'threshold': [0.001, 0.003, 0.005, 0.008],
        'estimator': {
            'sklearn.ensemble.ExtraTreesRegressor': {
                'n_estimators': [100],
                'max_features': np.arange(0.05, 0.5, 0.05)
            }
        },
        'max_features': [100, 200, 300]
    },

    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(1, 5),
        'score_func': {
            'sklearn.feature_selection.mutual_info_regression': None,
            'sklearn.feature_selection.f_regression': None
        }
    },
    # Preprocessors
    'sklearn.decomposition.FastICA': {
        'tol': np.arange(0.01, 1.01, 0.05)
    },

    'sklearn.cluster.FeatureAgglomeration': {
        'linkage': ['ward', 'complete', 'average'],
        'affinity': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    },

    'sklearn.preprocessing.Normalizer': {
        'norm': ['l1', 'l2', 'max']
    },

    'sklearn.kernel_approximation.Nystroem': {
        'kernel': ['rbf', 'cosine', 'chi2', 'laplacian', 'polynomial', 'poly', 'linear', 'additive_chi2', 'sigmoid'],
        'gamma': np.arange(0.01, 1.01, 0.05),
        'n_components': range(1, 11)
    },

    'sklearn.decomposition.PCA': {
        'svd_solver': ['randomized'],
        'iterated_power': range(1, 11)
    },

    'sklearn.preprocessing.PolynomialFeatures': {
        'degree': [2],
        'include_bias': [False],
        'interaction_only': [False]
    },

    'sklearn.kernel_approximation.RBFSampler': {
        'gamma': np.arange(0.01, 1.01, 0.05)
    },

    'sklearn.preprocessing.RobustScaler': {
    },

    'sklearn.preprocessing.StandardScaler': {
    },
    # regressor
    'sklearn.linear_model.ElasticNetCV': {
        'l1_ratio': np.arange(0.01, 1.01, 0.05),
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    },

    'sklearn.ensemble.ExtraTreesRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.ensemble.GradientBoostingRegressor': {
        'n_estimators': [100],
        'loss': ["ls", "lad", "huber", "quantile"],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'subsample': np.arange(0.05, 1.01, 0.05),
        'max_features': np.arange(0.05, 1.01, 0.05),
        'alpha': [0.75, 0.8, 0.85, 0.9, 0.95, 0.99]
    },

    'sklearn.ensemble.AdaBoostRegressor': {
        'n_estimators': [100],
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'loss': ["linear", "square", "exponential"]
    },

    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.linear_model.LassoLarsCV': {
        'normalize': [True, False]
    },

    'sklearn.svm.LinearSVR': {
        'loss': ["epsilon_insensitive", "squared_epsilon_insensitive"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1.]
    },

    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    },

    'sklearn.linear_model.RidgeCV': {
    },

    'xgboost.XGBRegressor': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1],
        'verbosity': [0],
        'objective': ['reg:squarederror']
    },

    'sklearn.linear_model.SGDRegressor': {
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive'],
        'penalty': ['elasticnet'],
        'alpha': [0.0, 0.01, 0.001],
        'learning_rate': ['invscaling', 'constant'],
        'fit_intercept': [True, False],
        'l1_ratio': [0.25, 0.0, 1.0, 0.75, 0.5],
        'eta0': [0.1, 1.0, 0.01],
        'power_t': [0.5, 0.0, 1.0, 0.1, 100.0, 10.0, 50.0]
    }
}

def get_models_dump(X_train, y_train):
    models_num = 1
    def dumps_exist():
        for i in range(0, models_num):
            if not os.path.exists(r'models_dump/model%d_dump' %i):
                return False
        else:
            return True

    def get_trained_dump():
        models = []
        for i in range(0, models_num):
            if not os.path.exists(r'models_dump/model%d_dump' % i):
                eval('exported_pipeline%d'%i).fit(X_train, y_train.iloc[:,i])
                with open('models_dump/model%d_dump'%i, 'wb') as f:
                    pickle.dump(eval('exported_pipeline%d'%i), f)
                print('model%d pickle saved' %i)
            models.append(eval('exported_pipeline%d'%i))
        return models

    if dumps_exist():
        models = []
        for i in range(0, models_num):
            with open('models_dump/model%d_dump'%i, 'rb') as f:
                models.append(pickle.load(f))
        print('models pickle loaded')
    else:
        models = get_trained_dump()
    return models
