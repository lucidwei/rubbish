from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor,GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectFwe, f_regression,SelectPercentile
from sklearn.linear_model import SGDRegressor, RidgeCV, LassoLarsCV, ElasticNetCV, ElasticNet
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator
from xgboost import XGBRegressor
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import Normalizer, StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.cluster import FeatureAgglomeration
import copy


exported_pipeline0 = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.3, min_samples_leaf=13, min_samples_split=12, n_estimators=100)),
    VarianceThreshold(threshold=0.05),
    PCA(iterated_power=8, svd_solver="randomized"),
    StackingEstimator(estimator=SGDRegressor(alpha=0.001, eta0=1.0, fit_intercept=True, l1_ratio=0.25, learning_rate="constant", loss="huber", penalty="elasticnet", power_t=10.0)),
    StackingEstimator(estimator=XGBRegressor(learning_rate=1.0, max_depth=10, min_child_weight=11, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.9500000000000001, verbosity=0)),
    StackingEstimator(estimator=SGDRegressor(alpha=0.01, eta0=1.0, fit_intercept=False, l1_ratio=0.0, learning_rate="invscaling", loss="huber", penalty="elasticnet", power_t=0.0)),
    StackingEstimator(estimator=XGBRegressor(learning_rate=1.0, max_depth=5, min_child_weight=7, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.7500000000000001, verbosity=0)),
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.8500000000000001, min_samples_leaf=4, min_samples_split=16, n_estimators=100)),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.5, loss="exponential", n_estimators=100)),
    StackingEstimator(estimator=SGDRegressor(alpha=0.001, eta0=1.0, fit_intercept=True, l1_ratio=0.25, learning_rate="constant", loss="huber", penalty="elasticnet", power_t=0.0)),
    LinearSVR(C=0.001, dual=True, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=0.1)
)

exported_pipeline1 = make_pipeline(
    StackingEstimator(estimator=RidgeCV()),
    Normalizer(norm="l2"),
    VarianceThreshold(threshold=0.01),
    RBFSampler(gamma=0.5),
    PCA(iterated_power=10, svd_solver="randomized"),
    SGDRegressor(alpha=0.01, eta0=0.1, fit_intercept=False, l1_ratio=0.0, learning_rate="constant", loss="huber", penalty="elasticnet", power_t=0.5)
)

exported_pipeline2 = make_pipeline(
    VarianceThreshold(threshold=0.05),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.01, max_depth=10, min_child_weight=8, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.9500000000000001, verbosity=0)),
    StackingEstimator(estimator=RidgeCV()),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.1, loss="ls", max_depth=4, max_features=0.05, min_samples_leaf=19, min_samples_split=20, n_estimators=100, subsample=0.7500000000000001)),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.1, loss="linear", n_estimators=100)),
    StackingEstimator(estimator=LassoLarsCV(normalize=True)),
    StackingEstimator(estimator=AdaBoostRegressor(learning_rate=0.001, loss="square", n_estimators=100)),
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.1, max_depth=4, min_child_weight=6, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=0.55, verbosity=0)),
    StandardScaler(),
    LinearSVR(C=0.001, dual=False, epsilon=0.01, loss="squared_epsilon_insensitive", tol=0.01)
)

exported_pipeline3 = make_pipeline(
    StackingEstimator(estimator=ExtraTreesRegressor(bootstrap=False, max_features=0.6500000000000001, min_samples_leaf=6, min_samples_split=11, n_estimators=100)),
    StandardScaler(),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.05, min_samples_leaf=4, min_samples_split=16, n_estimators=100)),
    SelectFwe(score_func=f_regression, alpha=0.003),
    RandomForestRegressor(bootstrap=False, max_features=1.0, min_samples_leaf=6, min_samples_split=7, n_estimators=100)
)

exported_pipeline4 = make_pipeline(
    VarianceThreshold(threshold=0.05),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.1, min_samples_leaf=20, min_samples_split=4, n_estimators=100)),
    StackingEstimator(estimator=ElasticNetCV(l1_ratio=0.15000000000000002, tol=0.0001)),
    RandomForestRegressor(bootstrap=True, max_features=0.9000000000000001, min_samples_leaf=9, min_samples_split=7, n_estimators=100)
)

exported_pipeline5 = make_pipeline(
    VarianceThreshold(threshold=0.1),
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=5, min_samples_leaf=14, min_samples_split=2)),
    StackingEstimator(estimator=LassoLarsCV(normalize=False)),
    MinMaxScaler(),
    Normalizer(norm="l2"),
    AdaBoostRegressor(learning_rate=0.5, loss="exponential", n_estimators=100)
)

exported_pipeline6 = make_pipeline(
    StackingEstimator(estimator=DecisionTreeRegressor(max_depth=4, min_samples_leaf=18, min_samples_split=12)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=0.1, loss="quantile", max_depth=7, max_features=0.9000000000000001, min_samples_leaf=2, min_samples_split=9, n_estimators=100, subsample=0.45)),
    ExtraTreesRegressor(bootstrap=False, max_features=0.4, min_samples_leaf=4, min_samples_split=8, n_estimators=100)
)

exported_pipeline7 = make_pipeline(
    make_union(
        FunctionTransformer(copy.copy),
        FeatureAgglomeration(affinity="manhattan", linkage="average")
    ),
    SelectPercentile(score_func=f_regression, percentile=47),
    StackingEstimator(estimator=SGDRegressor(alpha=0.001, eta0=1.0, fit_intercept=False, l1_ratio=0.5, learning_rate="constant", loss="huber", penalty="elasticnet", power_t=1.0)),
    StandardScaler(),
    ExtraTreesRegressor(bootstrap=False, max_features=0.9500000000000001, min_samples_leaf=2, min_samples_split=7, n_estimators=100)
)

exported_pipeline8 = make_pipeline(
    StandardScaler(),
    StandardScaler(),
    VarianceThreshold(threshold=0.0005),
    StackingEstimator(estimator=RandomForestRegressor(bootstrap=True, max_features=0.1, min_samples_leaf=18, min_samples_split=14, n_estimators=100)),
    AdaBoostRegressor(learning_rate=0.1, loss="square", n_estimators=100)
)

exported_pipeline9 = make_pipeline(
    StackingEstimator(estimator=XGBRegressor(learning_rate=0.001, max_depth=5, min_child_weight=10, n_estimators=100, n_jobs=1, objective="reg:squarederror", subsample=1.0, verbosity=0)),
    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.5, loss="lad", max_depth=3, max_features=0.35000000000000003, min_samples_leaf=20, min_samples_split=2, n_estimators=100, subsample=0.05)),
    StackingEstimator(estimator=LinearSVR(C=0.0001, dual=True, epsilon=0.0001, loss="squared_epsilon_insensitive", tol=0.0001)),
    RandomForestRegressor(bootstrap=True, max_features=0.7500000000000001, min_samples_leaf=1, min_samples_split=7, n_estimators=100)
)

exported_pipeline_enet = make_pipeline(
    ElasticNet(random_state=1996)
)
# param_list_enet = [
#     {'C': 0.1, 'tol': 1e-3},
# ]