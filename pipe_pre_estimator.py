# coding=gbk
import copy

import numpy as np
import pandas as pd
import talib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion, make_union
# from tsfresh.transformers import RelevantFeatureAugmenter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression, r_regression, mutual_info_regression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, RidgeCV
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
        print('...initializing MacroFE')

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        print('...transforming MacroFE')
        # feature names��transform�ﴫ������
        X_df = pd.DataFrame(X)
        gen = pd.DataFrame(index=X_df.index)
        for col_ind, col in X_df.iteritems():
            gen_i = single_generator(col_ind, col, type='macro')
            gen = pd.concat([gen, gen_i], axis=1)
        # self.names_out = gen.columns
        return np.array(gen)

    def get_feature_names_out(self, input_features=None):
        names_out = []
        for i in input_features:
            gen = [i+'_macd', i+'_macdsignal', i+'_macdhist', i+'_mom10',
                   i+'_PPO', i+'_yoy', i+'_mom', i+'_rsi14', i+'_rsi6', i+'_ema12']
            names_out.extend(gen)
        return np.array(names_out)


# TODO: ���԰�high low�ȼӽ������������ɸ�������
class AssetFE(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        pass


def single_generator(col_ind, feature_ori, type: str):
    generated_df = pd.DataFrame(index=feature_ori.index)
    col_ind = str(col_ind)
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


# ��ʱû�ã��Ȳ�����macro��asset
def get_asset_columns():
    info = utils_eda.get_info()
    mask = np.append(np.zeros(len(info.index) - 11, dtype=bool), np.ones(10, dtype=bool))
    return mask


# ������ԭʼ���ݵ���mask
def get_ori_columns(X):
    return [utils_eda.is_ori_id(id) for id in X.columns]


# ct = ColumnTransformer([
#     ('macroFE', MacroFE(), ~get_asset_columns()),
#     ('assetFE', AssetFE(), get_asset_columns()),
# ])

class FindCorrelation(BaseEstimator, TransformerMixin):
    """
    Remove pairwise correlations beyond threshold.
    This is not 'exact': it does not recalculate correlation
    after each step, and is therefore less expensive.
    This works similarly to the R caret::findCorrelation function
    with exact = False
    """
    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Produce binary array for filtering columns in feature array.
        Remember to transpose the correlation matrix so is
        column major.
        Loop through columns in (n_features,n_features) correlation matrix.
        Determine rows where value is greater than threshold.
        For the candidate pairs, one must be removed. Determine which feature
        has the larger average correlation with all other features and remove it.
        Remember, matrix is symmetric so shift down by one row per column as
        iterate through.
        """
        self.correlated = np.zeros(X.shape[1], dtype=bool)
        self.corr_mat = np.corrcoef(X.T)
        abs_corr_mat = np.abs(self.corr_mat)

        for i, col in enumerate(abs_corr_mat.T):
            corr_rows = np.where(col[i+1:] > self.threshold)[0]
            avg_corr = np.mean(col)

            if len(corr_rows) > 0:
                for j in corr_rows:
                    if np.mean(abs_corr_mat.T[:, j]) > avg_corr:
                        self.correlated[j] = True
                    else:
                        self.correlated[i] = True

        return self

    def transform(self, X, y=None):
        """
        Mask the array with the features flagged for removal
        """
        return X.T[~self.correlated].T

    def get_feature_names_out(self, input_features=None):
        return input_features[~self.correlated]


from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV, RidgeClassifierCV, SGDClassifier, Perceptron

num_select = 40
select_40n = FeatureUnion([
    ('lr0', SelectFromModel(estimator=LinearRegression(), max_features=num_select)),
    ('ridge0', SelectFromModel(estimator=RidgeCV(), max_features=num_select)),
    ('sgd0', SelectFromModel(estimator=SGDRegressor(), max_features=num_select)),
    ('rf0', SelectFromModel(estimator=RandomForestRegressor(random_state=1996), max_features=num_select)),
    ('gbr0', SelectFromModel(estimator=GradientBoostingRegressor(random_state=1996), max_features=num_select)),
    # MI̫���ˣ�ռ����ʱ���һ�룬����ѡ�����ĺ���Ƚ����
    ('mi0', SelectKBest(score_func=mutual_info_regression, k=num_select))
])
select_40n_cls = FeatureUnion([
    ('lr0', SelectFromModel(estimator=LogisticRegressionCV(max_iter=1000, random_state=1996), max_features=num_select)),
    ('ridge0', SelectFromModel(estimator=RidgeClassifierCV(), max_features=num_select)),
    ('sgd0', SelectFromModel(estimator=SGDClassifier(random_state=1996), max_features=num_select)),
    ('rf0', SelectFromModel(estimator=RandomForestClassifier(random_state=1996), max_features=num_select)),
    ('gbr0', SelectFromModel(estimator=GradientBoostingClassifier(random_state=1996), max_features=num_select)),
    # MI̫���ˣ�ռ����ʱ���һ�룬����ѡ�����ĺ���Ƚ����
    ('mi0', SelectKBest(score_func=mutual_info_classif, k=num_select))
])

num_20 = 20
select_20n = FeatureUnion([
    ('lr1', SelectFromModel(estimator=LinearRegression(), max_features=num_20)),
    ('ridge1', SelectFromModel(estimator=RidgeCV(), max_features=num_20)),
    ('sgd1', SelectFromModel(estimator=SGDRegressor(), max_features=num_20)),
    ('rf1', SelectFromModel(estimator=RandomForestRegressor(random_state=1996), max_features=num_20)),
    ('gbr1', SelectFromModel(estimator=GradientBoostingRegressor(random_state=1996), max_features=num_20)),
    ('mi1', SelectKBest(score_func=mutual_info_regression, k=num_20))
])
select_20n_cls = FeatureUnion([
    ('lr1', SelectFromModel(estimator=LogisticRegressionCV(max_iter=1000, random_state=1996), max_features=num_20)),
    ('ridge1', SelectFromModel(estimator=RidgeClassifierCV(), max_features=num_20)),
    ('sgd1', SelectFromModel(estimator=SGDClassifier(random_state=1996), max_features=num_20)),
    ('rf1', SelectFromModel(estimator=RandomForestClassifier(random_state=1996), max_features=num_20)),
    ('gbr1', SelectFromModel(estimator=GradientBoostingClassifier(random_state=1996), max_features=num_20)),
    ('mi1', SelectKBest(score_func=mutual_info_classif, k=num_20))
])

###################���������pipeline����ʵ������̫�鷳����Ϊ�Զ����pipeline�����׷���sklearn��
# union1 = FeatureUnion([
#     ("pca", CustomPCA(n_components=2)),
#     ("svd", CustomTruncatedSVD(n_components=2)),
# ])
#
# pipe1 = Pipeline([
#     ('get_stationary', pipe_preproc.GetStationary()),
#     ("union1", union1)
# ])
#
# talibFE = ColumnTransformer([
#     ('talibFE', MacroFE(), get_ori_columns),
# ])
#
# # TODO: feature union�Զ���ʧ����
# union = FeatureUnion([("PCA", pipe1),
#                       ('origin_station', pipe_preproc.GetStationary()),
#                       ('talibFE', MacroFE()),
#                       ])
#
# from train_FE import use_lag_x
#
# FE_ppl = Pipeline([
#     # TODO: ������������δ���X�Ŀ�ֵ�ģ�ϣ����ֱ����������ȱʧ���ݵĿ�ֵ
#     ('features', union),
#     ('fillna', KNNImputer()),
#     ('scaler1', StandardScaler()),
#     ('select_40n', select_40n),
#     # TODO: �ⲽ֮ǰ����û�ˣ������������Լ��Ĵ��ݻ��ơ�
#     # �Զ����pipe�������sklearn pipe�����Ϊsk��֧�ֶ�y��transform��
#     # �Զ���ķ�������df�����Ǵ������û��������ndarray�����˲���
#     ('series_to_supervised', pipe_preproc.SeriesToSupervised(n_in=use_lag_x, n_out=1)),
# ])
################################################

# �����Ǵ�sklearn pipeline
union = FeatureUnion([
    ('self', FunctionTransformer(copy.deepcopy, feature_names_out="one-to-one")),
    ("pca", PCA(n_components=2)),
    ("svd", TruncatedSVD(n_components=2)),
    ('talibFE', MacroFE()),
])

ppl_reg = Pipeline([
    # ('fillna0', KNNImputer()),
    ('scaler0', StandardScaler()),
    ('select_40n', select_40n),
    ('delcorr0', FindCorrelation(0.95)),
    # ͨ�����Ӳ��Լ���ʹ�ò��Լ����õ�ѵ�����Ĳ��ֽ�������������feature
    ('feature_gen', union),
    ('fillna1', KNNImputer()),
    ('scaler1', StandardScaler()),
    ('select_20n', select_20n),
    ('delcorr1', FindCorrelation(0.95)),
])

ppl_cls = Pipeline([
    # ('fillna0', KNNImputer()),
    ('scaler0', StandardScaler()),
    ('select_40n', select_40n_cls),
    ('delcorr0', FindCorrelation(0.95)),
    # ͨ�����Ӳ��Լ���ʹ�ò��Լ����õ�ѵ�����Ĳ��ֽ�������������feature
    ('feature_gen', union),
    ('fillna1', KNNImputer()),
    ('scaler1', StandardScaler()),
    ('select_20n', select_20n_cls),
    ('delcorr1', FindCorrelation(0.95))
])

def if_bin_col(X):
    mask = []
    for col in range(X.shape[1]):
        a = X.iloc[:, col].to_numpy()
        mask.append(((a==0) | (a==1)).all())
    return np.array(mask)


def if_numer_col(X):
    mask = ~if_bin_col(X)
    if not (mask==1).all():
        print('��������pass through��transformed X')
    return mask

# ��֤supplements����ι��estimator
ppl_cls_sup = ColumnTransformer([
    ('pipe_cls', ppl_cls, if_numer_col),
], remainder='passthrough')

ppl_reg_sup = ColumnTransformer([
    ('pipe_reg', ppl_reg, if_numer_col),
], remainder='passthrough')