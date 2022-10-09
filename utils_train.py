# coding=gbk

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from tpot import TPOTRegressor, TPOTClassifier
import numpy as np
from joblib import Memory
import pickle, os.path
from os.path import abspath


##### Generalized training to get model pipelines
def generate_1_pipe_auto(if_class, X, y, generations, population_size, max_time_mins, cachedir, pipe_num=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.8, test_size=0.2,
                                                        shuffle=False)
    memory = Memory(location=cachedir, verbose=0)
    cv = TimeSeriesSplit()
    if not if_class:
        pipeline_optimizer = TPOTRegressor(generations=generations, population_size=population_size, cv=cv,
                                           scoring='r2',
                                           early_stop=20,
                                           max_time_mins=max_time_mins,
                                           memory=memory,
                                           warm_start=True,
                                           periodic_checkpoint_folder=abspath('../../Documents/tpot_checkpoint'),
                                           log_file=abspath('../../Documents/tpot_log/log' + str(pipe_num)),
                                           random_state=1996, verbosity=3)
    else:
        pipeline_optimizer = TPOTClassifier(generations=generations, population_size=population_size, cv=cv,
                                            early_stop=10,
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


#### 利用得到的pipelines训练得到可执行模型
def get_models_dump(X_train, y_train, pipe, version, force_train, model_name):
    import pipe_pre_estimator
    import copy

    dir = r'models_dump/' + version
    if not os.path.exists(dir):
        os.makedirs(dir)
    models_dir = dir + r'/LastTrain' + str(X_train.index[-1])
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if pipe == 'benchmark':
        import pipe_models_base
        prefix = 'pipe_models_base.'
    elif pipe == 'post_FE':
        import pipe_models_FE
        prefix = 'pipe_models_FE.'
    elif pipe == 'cls':
        import pipe_models_cls
        prefix = 'pipe_models_cls.'
    else:
        raise Exception('Specify right pipeline to get dump')

    models_num = len(y_train.columns)
    models = []
    for i in range(0, models_num):
        file_path = models_dir + r'/modeldump_asset' + str(i)
        yi = y_train.iloc[:, i].copy(deep=True)
        if not os.path.exists(file_path) or force_train:
            if pipe == 'cls':
                if model_name == 'separate':
                    whole_ppl = make_pipeline(
                        pipe_pre_estimator.FE_ppl_cls,
                        eval(prefix + 'exported_pipeline%d' % i)
                    )
                else:
                    whole_ppl = make_pipeline(
                        pipe_pre_estimator.FE_ppl_cls,
                        eval(prefix + 'exported_pipeline_' + model_name)
                    )
            elif pipe == 'benchmark':
                if model_name == 'separate':
                    whole_ppl = make_pipeline(
                        eval(prefix + 'exported_pipeline%d' % i)
                    )
                else:
                    whole_ppl = make_pipeline(
                        eval(prefix + 'exported_pipeline_' + model_name)
                    )
            elif pipe == 'post_FE':
                if model_name == 'separate':
                    whole_ppl = make_pipeline(
                        pipe_pre_estimator.FE_ppl,
                        eval(prefix + 'exported_pipeline%d' % i)
                    )
                else:
                    whole_ppl = make_pipeline(
                        pipe_pre_estimator.FE_ppl,
                        eval(prefix + 'exported_pipeline_' + model_name)
                    )
            whole_ppl.fit(X_train.copy(deep=True), yi)
            print('样本内score：', whole_ppl.score(X_train, yi))

            # 写入缓存
            with open(file_path, 'wb') as f:
                pickle.dump(whole_ppl, f)
            models.append(whole_ppl)
            print('model %d pickle saved and appended' % i)
        else:
            with open(file_path, 'rb') as f:
                models.append(pickle.load(f))
            print('model %d pickle loaded' % i)

    return models
