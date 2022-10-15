from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier, LogisticRegressionCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier
from copy import copy

exported_pipeline0 = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=4, n_estimators=100, n_jobs=1, subsample=0.6000000000000001, verbosity=0)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=9, max_features=0.8, min_samples_leaf=2, min_samples_split=4, n_estimators=100, subsample=0.8)
)

exported_pipeline1 = make_pipeline(
    StackingEstimator(estimator=LinearSVC(C=1.0, dual=True, loss="hinge", penalty="l2", tol=0.1)),
    MLPClassifier(alpha=0.001, learning_rate_init=0.01)
)
# 这个可以留着
exported_pipeline2 = make_pipeline(
    StandardScaler(),
    SGDClassifier(alpha=0.001, eta0=0.01, fit_intercept=True, l1_ratio=0.5, learning_rate="constant", loss="squared_hinge", penalty="elasticnet", power_t=1.0)
)

exported_pipeline3 = make_pipeline(
    make_union(
        make_union(
            FunctionTransformer(copy),
            FunctionTransformer(copy)
        ),
        FunctionTransformer(copy)
    ),
    MLPClassifier(alpha=0.0001, learning_rate_init=0.01)
)

exported_pipeline4 = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.5, max_depth=5, max_features=0.9500000000000001, min_samples_leaf=5, min_samples_split=16, n_estimators=100, subsample=1.0)),
    LinearSVC(C=0.01, dual=True, loss="hinge", penalty="l2", tol=0.1)
)

exported_pipeline5 = make_pipeline(
    StackingEstimator(estimator=SGDClassifier(alpha=0.0, eta0=1.0, fit_intercept=False, l1_ratio=0.5, learning_rate="invscaling", loss="hinge", penalty="elasticnet", power_t=1.0)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=1, max_features=0.7000000000000001, min_samples_leaf=3, min_samples_split=10, n_estimators=100, subsample=0.6000000000000001)
)

exported_pipeline6 = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        StackingEstimator(estimator=MLPClassifier(alpha=0.01, learning_rate_init=0.01))
    ),
    VarianceThreshold(threshold=0.1),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=0.8500000000000001, min_samples_leaf=8, min_samples_split=19, n_estimators=100, subsample=0.8500000000000001)
)
# 这个样本外表现好像是最好的
exported_pipeline7 = make_pipeline(
    StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.55, min_samples_leaf=2, min_samples_split=15, n_estimators=100)),
    SGDClassifier(alpha=0.01, eta0=0.1, fit_intercept=False, l1_ratio=1.0, learning_rate="invscaling", loss="log", penalty="elasticnet", power_t=0.5)
)
exported_pipeline8 = make_pipeline(
    VarianceThreshold(threshold=0.01),
    # OneHotEncoder(minimum_fraction=0.2, sparse=False, threshold=10),
    # OneHotEncoder(minimum_fraction=0.2, sparse=False, threshold=10),
    SGDClassifier(alpha=0.001, eta0=0.1, fit_intercept=False, l1_ratio=0.75, learning_rate="constant", loss="squared_hinge", penalty="elasticnet", power_t=100.0)
)
exported_pipeline9 = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=4, max_features=0.4, min_samples_leaf=7, min_samples_split=16, n_estimators=100, subsample=0.9000000000000001)),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="entropy", max_depth=3, min_samples_leaf=3, min_samples_split=12)),
    StackingEstimator(estimator=DecisionTreeClassifier(criterion="gini", max_depth=7, min_samples_leaf=14, min_samples_split=18)),
    MLPClassifier(alpha=0.0001, learning_rate_init=0.1)
)

# exported_pipelineX = make_pipeline(
#     StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.55, min_samples_leaf=2, min_samples_split=15, n_estimators=100)),
#     SGDClassifier(alpha=0.01, eta0=0.1, fit_intercept=False, l1_ratio=1.0, learning_rate="invscaling", loss="log", penalty="elasticnet", power_t=0.5)
# )

# 适合pipe9 -0.52
# exported_pipeline_sgd = make_pipeline(
#     SGDClassifier(alpha=0.001, eta0=0.01, fit_intercept=True, l1_ratio=0.5, shuffle=False, learning_rate="adaptive", loss="modified_huber", penalty="elasticnet", power_t=1.0, average=10)
# )
# 再pretrain from log9
exported_pipeline_sgd9 = make_pipeline(
    SGDClassifier(alpha=0.0, eta0=0.1, fit_intercept=False, l1_ratio=0.75, learning_rate="constant", loss="squared_hinge", penalty="elasticnet", power_t=1.0, average=10)
)
# 再pretrain from log3
exported_pipeline_sgd3 = make_pipeline(
    SGDClassifier(alpha=0.0, eta0=0.1, fit_intercept=True, l1_ratio=0.75, learning_rate="constant", loss="modified_huber", penalty="elasticnet", power_t=10, average=10)
)
# 再pretrain from log6
exported_pipeline_sgd6 = make_pipeline(
    SGDClassifier(alpha=0.01, eta0=0.01, fit_intercept=False, l1_ratio=0.75, learning_rate="invscaling", loss="modified_huber", penalty="elasticnet", power_t=0.5, average=10)
)
# 再pretrain from log2
exported_pipeline_sgd2 = make_pipeline(
    SGDClassifier(alpha=0.01, eta0=0.01, fit_intercept=True, l1_ratio=1, learning_rate="constant", loss="log", penalty="elasticnet", power_t=0.1, average=10)
)
# 再pretrain from log5
exported_pipeline_sgd5 = make_pipeline(
    SGDClassifier(alpha=0.001, eta0=1, fit_intercept=False, l1_ratio=1, learning_rate="constant", loss="perceptron", penalty="elasticnet", power_t=1, average=10)
)
# 再pretrain from log4
exported_pipeline_sgd4 = make_pipeline(
    SGDClassifier(alpha=0.001, eta0=0.1, fit_intercept=False, l1_ratio=1, learning_rate="constant", loss="modified_huber", penalty="elasticnet", power_t=0.5, average=10)
)
# 再pretrain from log1
exported_pipeline_sgd1 = make_pipeline(
    SGDClassifier(alpha=0.01, eta0=0.1, fit_intercept=True, l1_ratio=1, learning_rate="invscaling", loss="log", penalty="elasticnet", power_t=1, average=10)
)

# 适合pipe179- 0.44，42，60
# exported_pipeline_svc = make_pipeline(
#     LinearSVC(C=0.0001, dual=False, loss="squared_hinge", penalty="l2", tol=0.001)
# )
# 再pretrain from log8
exported_pipeline_svc8 = make_pipeline(
    LinearSVC(C=0.0001, dual=True, loss="hinge", penalty="l2", tol=0.00001)
)
# 再pretrain from log3
exported_pipeline_svc3 = make_pipeline(
    LinearSVC(C=0.001, dual=True, loss="squared_hinge", penalty="l2", tol=0.001)
)

exported_pipeline_svccv = make_pipeline(
    LinearSVC()
)

# 再pretrain from log8
exported_pipeline_gb8 = make_pipeline(
    GradientBoostingClassifier(learning_rate=0.5, max_depth=4, max_features=0.05, min_samples_leaf=16, min_samples_split=4, n_estimators=100, subsample=0.6)
)
# 再pretrain from log6
exported_pipeline_gb6 = make_pipeline(
    GradientBoostingClassifier(learning_rate=0.1, max_depth=3, max_features=0.75, min_samples_leaf=2, min_samples_split=3, n_estimators=100, subsample=0.55)
)
# 再pretrain from log1
exported_pipeline_gb1 = make_pipeline(
    GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=0.05, min_samples_leaf=5, min_samples_split=11, n_estimators=100, subsample=0.9)
)

exported_pipeline_gb = make_pipeline(
    GradientBoostingClassifier(learning_rate=0.1, max_depth=2, max_features=0.75, min_samples_leaf=0.3, min_samples_split=0.6, n_estimators=200, subsample=0.9)
)

exported_pipeline_gbcv = make_pipeline(
    GradientBoostingClassifier(learning_rate=0.02, n_estimators=200, min_samples_leaf=0.3, min_samples_split=0.7, max_depth=2, max_features=0.6)
)

# 适合1679 - 0.47, 43, 46, 69
# model appendging bugx修改后：
# 适合2 - 0.43
exported_pipeline_rf = make_pipeline(
    RandomForestClassifier(n_estimators=300, max_depth=3, bootstrap=False, random_state=1996)
)
# 再pretrain from log7&5
exported_pipeline_mlp7 = make_pipeline(
    MLPClassifier(alpha=0.1, learning_rate_init=0.001)
)
# 再pretrain from log0
exported_pipeline_mlp0 = make_pipeline(
    MLPClassifier(alpha=0.01, learning_rate_init=0.5)
)

# 适合167 - 0.48 48 51
# 026 - 0.42 46 42
exported_pipeline_stksvc = make_pipeline(
    StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.1, max_depth=4, max_features=0.4,
                                                           min_samples_leaf=7, min_samples_split=16, n_estimators=100, subsample=0.9000000000000001)),
    StackingEstimator(estimator=RandomForestClassifier(criterion="entropy", max_depth=3, min_samples_leaf=3, min_samples_split=12)),
    StackingEstimator(estimator=MLPClassifier(alpha=0.0001, learning_rate_init=0.1)),
    StackingEstimator(estimator=SGDClassifier(alpha=0.0, eta0=1.0, fit_intercept=False, l1_ratio=0.5, learning_rate="invscaling",
                                              loss="hinge", penalty="elasticnet", power_t=1.0)),
    StackingEstimator(
        estimator=XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=4, n_estimators=100, n_jobs=1,
                                subsample=0.6000000000000001, verbosity=0)),
    LinearSVC(C=0.01, dual=True, loss="hinge", penalty="l2", tol=0.1)
)


exported_pipeline_logitcv = make_pipeline(
    LogisticRegressionCV(cv=TimeSeriesSplit(), max_iter=1000, fit_intercept=True)
)

exported_pipeline_logit = make_pipeline(
    LogisticRegression(max_iter=1000, fit_intercept=True)
)

param_list_logit = [
    {'C': 0.1, 'tol': 1e-3},
    {'C': 0.1, 'tol': 1e-3},
    {'C': 0.01, 'tol': 1e-4},
    {'C': 1e-3, 'tol': 1e-3},
    {'C': 1e-3, 'tol': 1e-3},
    {'C': 1e-2, 'tol': 1e-4},
    {'C': 1e-1, 'tol': 1e-2},
    {'C': 0.046, 'tol': 1e-3},
    {'C': 0.046, 'tol': 1e-2},
    {'C': 1e-2, 'tol': 1e-4},
]

