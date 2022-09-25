from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, OneHotEncoder
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
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
# 这个样本外表现好像是最好的
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

exported_pipelineX = make_pipeline(
    StandardScaler(),
    SGDClassifier(alpha=0.001, eta0=0.01, fit_intercept=True, l1_ratio=0.5, learning_rate="constant", loss="squared_hinge", penalty="elasticnet", power_t=1.0)
)