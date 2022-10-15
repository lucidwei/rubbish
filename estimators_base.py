import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.svm import LinearSVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel, SelectPercentile, mutual_info_regression
from sklearn.decomposition import PCA
from sklearn.cluster import FeatureAgglomeration
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MaxAbsScaler, RobustScaler, StandardScaler, Normalizer, Binarizer, PolynomialFeatures


# 从1复制的
exported_pipeline0 = make_pipeline(
    SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.45, n_estimators=100), threshold=0.002),
    MaxAbsScaler(),
    GradientBoostingRegressor(alpha=0.85, learning_rate=0.001, loss="ls", max_depth=8, max_features=0.05, min_samples_leaf=15, min_samples_split=3, n_estimators=100, subsample=0.4)
)

exported_pipeline1 = make_pipeline(
    SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.45, n_estimators=100), threshold=0.002),
    MaxAbsScaler(),
    GradientBoostingRegressor(alpha=0.85, learning_rate=0.001, loss="ls", max_depth=8, max_features=0.05, min_samples_leaf=15, min_samples_split=3, n_estimators=100, subsample=0.4)
)

exported_pipeline2 = make_pipeline(
    SelectPercentile(score_func=mutual_info_regression, percentile=11),
    RobustScaler(),
    ExtraTreesRegressor(bootstrap=False, max_features=0.7000000000000001, min_samples_leaf=2, min_samples_split=12, n_estimators=100)
)

exported_pipeline3 = make_pipeline(
    SelectPercentile(score_func=mutual_info_regression, percentile=15),
    StandardScaler(),
    LassoLarsCV(normalize=False)
)

exported_pipeline4 = make_pipeline(
    SelectPercentile(score_func=mutual_info_regression, percentile=5),
    Normalizer(norm="l2"),
    LassoLarsCV(normalize=True)
)

exported_pipeline5 = make_pipeline(
    SelectPercentile(score_func=mutual_info_regression, percentile=23),
    PCA(iterated_power=6, svd_solver="randomized"),
    ExtraTreesRegressor(bootstrap=False, max_features=1.0, min_samples_leaf=11, min_samples_split=10, n_estimators=100)
)

exported_pipeline6 = make_pipeline(
    SelectPercentile(score_func=mutual_info_regression, percentile=6),
    FeatureAgglomeration(affinity="l2", linkage="average"),
    LassoLarsCV(normalize=False)
)

exported_pipeline7 = make_pipeline(
    SelectPercentile(score_func=mutual_info_regression, percentile=10),
    Binarizer(threshold=0.11), # 这个就很奇怪
    LinearSVR(C=0.0001, dual=True, epsilon=0.1, loss="squared_epsilon_insensitive", tol=0.0001)
)

exported_pipeline8 = make_pipeline(
    SelectPercentile(score_func=mutual_info_regression, percentile=29),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    ExtraTreesRegressor(bootstrap=True, max_features=0.05, min_samples_leaf=16, min_samples_split=4, n_estimators=100)
)

exported_pipeline9 = make_pipeline(
    SelectPercentile(score_func=mutual_info_regression, percentile=23),
    RobustScaler(),
    LassoLarsCV(normalize=True)
)