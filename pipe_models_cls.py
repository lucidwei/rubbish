from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
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

#######
exported_pipeline7 = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=4, n_estimators=100, n_jobs=1, subsample=0.6000000000000001, verbosity=0)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=9, max_features=0.8, min_samples_leaf=2, min_samples_split=4, n_estimators=100, subsample=0.8)
)
exported_pipeline8 = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=4, n_estimators=100, n_jobs=1, subsample=0.6000000000000001, verbosity=0)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=9, max_features=0.8, min_samples_leaf=2, min_samples_split=4, n_estimators=100, subsample=0.8)
)
exported_pipeline9 = make_pipeline(
    StackingEstimator(estimator=XGBClassifier(learning_rate=0.1, max_depth=6, min_child_weight=4, n_estimators=100, n_jobs=1, subsample=0.6000000000000001, verbosity=0)),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=9, max_features=0.8, min_samples_leaf=2, min_samples_split=4, n_estimators=100, subsample=0.8)
)
