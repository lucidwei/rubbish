# coding=gbk

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle, datetime

import utils
import pandas as pd
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoLarsCV
from tpot.builtins import StackingEstimator

# Configuration
PATH_ORI_DATA = r'C:\Users\lucid\Documents\长江实习\课题之自上而下\data'
## 原始数据文件是否已经更新
if_update = False
## 是否使用缓存的数据
use_cache = False
align_to = 'month'
begT = '2005-01'
endT = datetime.date.today()


X, y = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, align_to, begT, endT)

X_train, X_test, y_train, y_test = train_test_split(X, y.iloc[:, 0],
                                                    train_size=0.75, test_size=0.25,
                                                    shuffle=False)

pipeline_optimizer = TPOTRegressor(generations=100, population_size=100, cv=5,
                                    random_state=1996, verbosity=2)

# pipeline_optimizer = make_pipeline(
#     SelectFromModel(estimator=ExtraTreesRegressor(max_features=0.8500000000000001, n_estimators=100), threshold=0.45),
#     StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.8, learning_rate=0.01, loss="quantile", max_depth=4,
#                                                           max_features=0.1, min_samples_leaf=14, min_samples_split=18,
#                                                           n_estimators=100, subsample=0.15000000000000002)),
#     LassoLarsCV(normalize=False)
# )

pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pred = pipeline_optimizer.predict(X_test)
# print(pred)
pipeline_optimizer.export('./tpot_gen/trial_pipeline1.py')
