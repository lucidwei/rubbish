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
## 预处理逻辑(参数)变更/缓存的pickle需要更新时，设为False
use_cache = False
## 预处理参数
align_to = 'month'
use_lag_x = 15
begT = '2004-01'
endT = datetime.date.today()

X, y = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, align_to, use_lag_x, begT, endT)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.75, test_size=0.25,
                                                    shuffle=False)

models = utils.get_models_dump(X_train, y_train)
evluator = Evaluator(models, X_test, y_test)

evaluator.plot_port_return()
evaluator.plot_excess_return()

pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pred = pipeline_optimizer.predict(X_test)
# print(pred)
pipeline_optimizer.export('./tpot_gen/trial_pipeline1.py')
