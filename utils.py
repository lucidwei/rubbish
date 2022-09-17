# coding=gbk
# lucid   ��ǰϵͳ�û�
# 2022/8/16   ��ǰϵͳ����
# 14:47   ��ǰϵͳʱ��
# PyCharm   �����ļ���IDE����
import pickle, os.path, platform
from os.path import abspath
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from tpot import TPOTRegressor
from joblib import Memory
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
        'prepipe_data': r'data_dump\prepipe_data'
    }

    mac_dic = {
        'data_x': './data/data_x.csv',
        'data_y': './data/data_y.csv',
        'to_month_table': './data/to_month_table.xlsx',
        'to_week_table': './data/to_week_table.xlsx',
        'info_table': './data/info_table.csv',
        'prepipe_data': './data_dump/prepipe_data'
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

        # ����Ԥ����
        info = pd.read_excel(info_path, index_col=0, engine="openpyxl")
        pipe_preprocess = Pipeline(steps=[
            ('special_treatment', pipe_preproc.SpecialTreatment(info)),
            ('data_alignment', pipe_preproc.DataAlignment(align_to, info)),
            ('station_origin', pipe_preproc.GetStationary()),
            ('ts_to_supervised', pipe_preproc.SeriesToSupervised(n_in=use_x_lags))
        ])

        X, y = pipe_preprocess.fit_transform(raw_x, raw_y)
        # ������̬������bfill����X��selectFromModel��y�����п�
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


def generate_1_pipe_auto(X, y, generations, population_size, max_time_mins, cachedir, pipe_num=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8, test_size=0.2,
                                                        shuffle=False)
    memory = Memory(location=cachedir, verbose=0)
    cv = TimeSeriesSplit()
    pipeline_optimizer = TPOTRegressor(generations=generations, population_size=population_size, cv=cv,
                                       scoring='r2',
                                       early_stop=20,
                                       max_time_mins=max_time_mins,
                                       memory=memory,
                                       warm_start=True,
                                       periodic_checkpoint_folder=abspath('../../Documents/tpot_checkpoint'),
                                       log_file=abspath('../../Documents/tpot_log/log' + str(pipe_num)),
                                       random_state=1996, verbosity=3)
    pipeline_optimizer.fit(X_train, y_train)
    print('A pipe finised, score(X_test, y_test):', pipeline_optimizer.score(X_test, y_test))
    if pipe_num is None:
        pipeline_optimizer.export('./tpot_gen/multioutput_tpotpipe.py')
    else:
        pipeline_optimizer.export('./tpot_gen/separate_tpotpipe%d.py' % pipe_num)

    return pipeline_optimizer, X_test, y_test


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
                                       log_file=abspath('../../Documents/tpot_log/log' + str(pipe_num)),
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


def get_models_dump(X_train, y_train, version):
    import pipe_FE
    import copy

    models_num = len(y_train.columns)
    models = []
    for i in range(0, models_num):
        if version == 'benchmark':
            import pipe_models_base
            prefix = 'pipe_models_base.'
            file_path = r'models_dump/benchmark/model%d_dump' % i
        elif version == 'post_FE':
            import pipe_models_FE
            prefix = 'pipe_models_FE.'
            file_path = r'models_dump/post_FE/model%d_dump' % i

            # if not os.path.exists(file_path):
                # �����㲻�У�û���õ��������ݡ��������ݽ�ά�����ϱ���ѵ������һ��transform��
                # TODO����Ȼ��δ�����ݵ����ɣ��������ܽ�ʡ�ܶ�transform��ʱ�䡣����Ŀǰ�һ���֪��ΰ�y�ŵ�ppl��
                # ����������trainѵ������Ȼ����Լ���fitted��pipelineֱ��transform������fit��ôд���в����
                # ����������get_stationary�Ż�Ԥ�����to_supervised�ŵ�ppl��
                # print('feature engineering before training')
                # X_supervised, yi = pipe_FE.FE_ppl.fit_transform(copy.deepcopy(X_train), y_train.iloc[:, i])
                # X_selected = pipe_FE.select_20n.fit_transform(X_supervised, yi)
                # print('feature engineering finished')
        else:
            raise Exception('Please specify the right version of models to get')

        yi = y_train.iloc[:, i].copy(deep=True)

        if not os.path.exists(file_path):
            whole_ppl = make_pipeline(
                pipe_FE.FE_ppl,
                eval(prefix + 'exported_pipeline%d' % i)
            )
            whole_ppl.fit(X_train.copy(deep=True), yi)
            with open(file_path, 'wb') as f:
                pickle.dump(whole_ppl, f)
            models.append(whole_ppl)
            print('model %d pickle saved and appended' % i)
        else:
            with open(file_path, 'rb') as f:
                models.append(pickle.load(f))
            print('model %d pickle loaded' % i)

    return models

# �������Ӧ���Ǵ򲻳��ˣ������������̰�
# def get_models_dump_patch(X_train, y_train, version):
#     import pipe_FE
#     import copy
#     models_num = len(y_train.columns)
#     models = []
#     for i in range(0, models_num):
#         if version == 'benchmark':
#             file_path = r'models_dump/benchmark/model%d_dump' % i
#             yi = y_train.iloc[:, i]
#             # TODO��deprecated, ������������
#             X_selected = X_train.copy(deep=True)
#         elif version == 'post_FE':
#             file_path = r'models_dump/post_FE/model%d_dump' % i
#             if not os.path.exists(file_path):
#                 # TODO����Ȼ��δ�����ݵ����ɣ��������ܽ�ʡ�ܶ�transform��ʱ�䡣����Ŀǰ�һ���֪��ΰ�y�ŵ�ppl��
#                 # ����������trainѵ������Ȼ����Լ���fitted��pipelineֱ��transform������fit��ôд���в����
#                 print('feature engineering before training')
#                 X_supervised, yi = pipe_FE.FE_ppl.fit_transform(copy.deepcopy(X_train), y_train.iloc[:, i])
#                 X_selected = pipe_FE.select_20n.fit_transform(X_supervised, yi)
#                 print('feature engineering finished')
#         else:
#             raise Exception('Please specify the right version of models to get')
#
#         if not os.path.exists(file_path):
#             X_train, X_test, y_train, y_test = train_test_split(X_selected, yi,
#                                                                 train_size=0.8, test_size=0.2,
#                                                                 shuffle=False)
#             eval('exported_pipeline%d' % i).fit(X_train, y_train)
#             with open(file_path, 'wb') as f:
#                 pickle.dump(eval('exported_pipeline%d' % i), f)
#             models.append(eval('exported_pipeline%d' % i))
#             print('model %d pickle saved and appended' % i)
#         else:
#             with open(file_path, 'rb') as f:
#                 models.append(pickle.load(f))
#             print('model %d pickle loaded' % i)
#
#     return models, X_test, y_test, X_train, y_train # ���Ҳ���а������ص��ǵ���ģ�͵�����


# ������benchָ��Ȩ��ϣ�����benchmark model��Ҫ��model list��Ϊ��������
class Evaluator:
    def __init__(self, models, X_test, y_test, X_train, y_train):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        # ѵ������ֵ��position��Ҫtrain data
        self.X_train = X_train
        self.y_train = y_train
        self.port_pos = self.get_port_pos()
        self.port_ret = self.get_port_ret()
        self.bench_ret = self.get_bench_ret()

    # ����z-score��������percentile�����λ
    def get_port_pos(self):
        # ��y_train��z-scoreʱ�õ���ֵ��׼������pred��y_test normalize
        pos_info = pd.DataFrame(columns=['avg', 'std'], index=self.y_train.columns)
        for col_ind, col in self.y_train.iteritems():
            pos_info.loc[col_ind, 'avg'] = np.average(col)
            pos_info.loc[col_ind, 'std'] = np.std(col)
        # ��Ԥ��������ת��Ϊz-score
        pos_z = pd.DataFrame(index=self.y_test.index, columns=self.y_test.columns)
        i = 0
        for col_ind in self.y_test.columns:
            pred = self.models[i].predict(self.X_test)
            pos_z.iloc[:, i] = (pred - pos_info.loc[col_ind, 'avg']) / pos_info.loc[col_ind, 'std']
            i += 1
        # z-scoreת��Ϊposition
        pos = pd.DataFrame(index=pos_z.index, columns=pos_z.columns)
        for row_ind, row in pos_z.iterrows():
            row[row < 0] = 0  # ����������
            pos.loc[row_ind, :] = row / sum(row)

        return pos

    def get_port_ret(self):
        ret_df = pd.DataFrame(index=self.y_test.index, columns=['return'])
        for i in self.y_test.index:
            ret_df.loc[i, 'return'] = np.average(self.y_test.loc[i, :], weights=self.port_pos.loc[i, :])
        return ret_df

    # ��Ȩ���
    def get_bench_ret(self):
        weights = [1 / len(self.y_test.columns) for _ in self.y_test.columns]
        # TODO: ����ծ��returnӦ��ȡ�෴�������������ƱϢ
        ret_df = pd.DataFrame(index=self.y_test.index, columns=['return'])
        for i in self.y_test.index:
            ret_df.loc[i, 'return'] = np.average(self.y_test.loc[i, :], weights=weights)
        # TODO: ��������ʵ���Ƿ�һ�£���ô�о���̫С��
        return ret_df

    def get_port_worth(self):
        return return_to_worth(self.port_ret)

    def get_bench_worth(self):
        return return_to_worth(self.bench_ret)


# ��ĩ��ֵ
def return_to_worth(ret_df):
    worth_df = pd.DataFrame(index=ret_df.index, columns=['worth'])
    for i in range(len(worth_df.index)):
        if i == 0:
            worth_df.iloc[i, 0] = 1 + ret_df.iloc[i, 0]
        else:
            worth_df.iloc[i, 0] = (1 + ret_df.iloc[i, 0]) * worth_df.iloc[i - 1, 0]
    return worth_df


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
