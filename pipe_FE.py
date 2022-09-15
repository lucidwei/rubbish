# coding=gbk
import copy

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
# from tsfresh.transformers import RelevantFeatureAugmenter
from sklearn.base import BaseEstimator, TransformerMixin
import pickle, talib
import utils_eda


# TODO�����̣�get_stationary��transformcolumn(get_tsfresh, get_talib), (���������ǲ��е�featureunion)
#  Ȼ��series_to_supervised�����feature selectionѡ��100��

# get_stationary����õĻ����Ը��ƹ���

# ColumnTransformer

# ����һ��toy���ݶ��е��������ҽ����Բ�ǿ���Ȳ�����
# class TsPrepare(BaseEstimator, TransformerMixin):
#     def __init__(self, backward:bool):
#         self.backward = backward
#         print('transforming data for feature augmenter')
#
#     # ���ö�y����
#     def fit(self, X, y=None):
#         return self
#
#     def transform(self, X, y=None):
#         if self.backward = False:
#             # ����һ��id=1��indexʱ��תΪtime��
#             pass
#         else:
#             pass

# GetTsfresh �ɽ����Բ��ߣ�������Ϊ�������䡣
# pipeline = Pipeline([
#             # ('ts_prepare'),
#             ('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
#             ('classifier', RandomForestClassifier()),
#             ])
# X = pd.DataFrame(index=y.index)


# GetTailb
# ģ��preproc�����������ݣ���talib���϶��������ֵ��ȥ��ֵ����series_to_supervised��������
class MacroFE(BaseEstimator, TransformerMixin):
    def __init__(self):
        print('...initializing MacroFE\n')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('...transforming MacroFE\n')
        gen = pd.DataFrame(index=X.index)
        for col_ind, col in X.iteritems():
            gen_i = single_generator(col_ind, col, type='macro')
            gen = pd.concat([gen, gen_i], axis=1)
        return gen


# TODO: ���԰�high low�ȼӽ������������ɸ�������
class AssetFE(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pass


def single_generator(col_ind, feature_ori, type: str):
    generated_df = pd.DataFrame(index=feature_ori.index)
    generated_df[col_ind + '_macd'], generated_df[col_ind + '_macdsignal'], generated_df[
        col_ind + '_macdhist '] = talib.MACD(feature_ori)
    generated_df[col_ind + '_mom10'] = talib.MOM(feature_ori)
    generated_df[col_ind + '_PPO'] = talib.PPO(feature_ori)
    generated_df[col_ind + '_yoy'] = talib.ROCP(feature_ori, timeperiod=12)
    generated_df[col_ind + '_mom'] = talib.ROCP(feature_ori, timeperiod=1)
    generated_df[col_ind + '_rsi14'] = talib.RSI(feature_ori, timeperiod=14)
    generated_df[col_ind + '_rsi6'] = talib.RSI(feature_ori, timeperiod=6)
    generated_df[col_ind + '_ema12'] = talib.EMA(feature_ori, timeperiod=12)
    return generated_df


# ��ʱ���ã��Ȳ�����macro��asset
def get_asset_columns():
    info = utils_eda.get_info()
    mask = np.append(np.zeros(len(info.index) - 11, dtype=bool), np.ones(10, dtype=bool))
    return mask


# ������ԭʼ���ݵ���mask
def get_ori_columns(X):
    return [utils_eda.is_ori_id(id) for id in X.columns]


##################################�������Խ׶δ���
with open('data_dump/prepipe_data', 'rb') as f:
    (X, y) = pickle.load(f)
print('data pickle loaded')
use_lag_x = 10

# ct = ColumnTransformer([
#     ('macroFE', MacroFE(), ~get_asset_columns()),
#     ('assetFE', AssetFE(), get_asset_columns()),
# ])


from sklearn.decomposition import PCA, TruncatedSVD
import pipe_preproc
from math import log, sqrt


# def add_col_name(func, col_list):
#     '''Decorator that returns DF with column names.'''
#     def wrap(*args, **kwargs):
#         result = func(*args, **kwargs)
#         df = pd.DataFrame(result, index=X.index, columns=col_list)
#         return df
#     return wrap

# �Զ���һ�°������Ż�ȥ
class CustomPCA(PCA):
    def fit_transform(self, X, y=None):
        U = super().fit_transform(X)
        df = pd.DataFrame(U, index=X.index, columns=['PC1', 'PC2'])
        return df


class CustomTruncatedSVD(TruncatedSVD):
    def fit_transform(self, X, y=None):
        U = super().fit_transform(X)
        df = pd.DataFrame(U, index=X.index, columns=['SVD1', 'SVD2'])
        return df


union1 = FeatureUnion([
    ("pca", CustomPCA(n_components=2)),
    ("svd", CustomTruncatedSVD(n_components=2)),
])

pipe1 = Pipeline([
    ('get_stationary', pipe_preproc.GetStationary()),
    ("union1", union1)
])

talibFE = ColumnTransformer([
    ('talibFE', MacroFE(), get_ori_columns),
])

union = FeatureUnion([("PCA", pipe1),
                      #TODO: �ǵý��ע��
                      # ('origin_station', pipe_preproc.GetStationary()),
                      ('talibFE', talibFE),
                      ])

from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, r_regression, mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV
from sklearn.metrics import r2_score
import utils

num_select = 40
select_40n = FeatureUnion([
    ('lr', SelectFromModel(estimator=LinearRegression(), max_features=num_select)),
    ('ridge', SelectFromModel(estimator=RidgeCV(), max_features=num_select)),
    ('sgd', SelectFromModel(estimator=SGDRegressor(), max_features=num_select)),
    ('rf', SelectFromModel(estimator=RandomForestRegressor(random_state=1996), max_features=num_select)),
    ('gbr', SelectFromModel(estimator=GradientBoostingRegressor(random_state=1996), max_features=num_select)),
    # MI̫���ˣ�ռ����ʱ���һ�룬����ѡ�����ĺ���Ƚ����
    ('mi', SelectKBest(score_func=mutual_info_regression, k=num_select))
])

num_select = 20
select_20n = FeatureUnion([
    ('lr', SelectFromModel(estimator=LinearRegression(), max_features=num_select)),
    ('ridge', SelectFromModel(estimator=RidgeCV(), max_features=num_select)),
    ('sgd', SelectFromModel(estimator=SGDRegressor(), max_features=num_select)),
    ('rf', SelectFromModel(estimator=RandomForestRegressor(random_state=1996), max_features=num_select)),
    ('gbr', SelectFromModel(estimator=GradientBoostingRegressor(random_state=1996), max_features=num_select)),
    # MI̫���ˣ�ռ����ʱ���һ�룬����ѡ�����ĺ���Ƚ����
    ('mi', SelectKBest(score_func=mutual_info_regression, k=num_select))
])

# select_40n = FeatureUnion([
#     ('lr', SelectFromModel(estimator=LinearRegression())),
#     ('ridge', SelectFromModel(estimator=RidgeCV())),
#     ('sgd', SelectFromModel(estimator=SGDRegressor())),
#     ('rf', SelectFromModel(estimator=RandomForestRegressor(random_state=1996))),
#     ('gbr', SelectFromModel(estimator=GradientBoostingRegressor(random_state=1996))),
#     # MI̫���ˣ�ռ����ʱ���һ�룬����ѡ�����ĺ���Ƚ����
#     ('mi', SelectKBest(score_func=mutual_info_regression, k=num_select))
# ])
# select_40n.set_params(max_features=40)

ppl = Pipeline([
    ('features', union),
    ('fillna', SimpleImputer(strategy='mean')),
    ('selector1', select_40n),
    ('series_to_supervised', pipe_preproc.SeriesToSupervised(n_in=use_lag_x, n_out=1)),
])


i = 0  # ����Ϊѭ��ѵ�����
for yi_ind, yi in y.iloc[:, i:].iteritems():
    X_supervised = ppl.fit_transform(X, yi)
    X_selected = select_20n.fit_transform(X_supervised, yi)
    pipe, X_test, y_test = utils.generate_1_pipe_light(X, yi, generations, population_size, max_time_mins, cachedir,
                                                       pipe_num=i, tpot_config=utils.tpot_config)
    preds = pipe.predict(X_test)
    print('��%d���ʲ���Pipe r2 score:' % i, r2_score(y_test, preds))
    i += 1
##################################
