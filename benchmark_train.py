# coding=gbk

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle, datetime, platform
import pipe_preproc
import utils
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor
from tempfile import mkdtemp
from joblib import Memory
from shutil import rmtree

# Configuration
PATH_ORI_DATA = r'C:\Users\lucid\Documents\长江实习\课题之自上而下\data'
## 原始数据文件是否已经更新
if_update = False
## 预处理逻辑(参数)变更/缓存的pickle需要更新时，设为False
use_cache = True
## 预处理参数
align_to = 'month'
use_lag_x = 15
begT = '2004-01'
endT = datetime.date.today()
## 训练参数
### 对于y的multi-output，是否分开训练
if_separate_y = True
generations = 1
population_size = 2

X, y = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, align_to, use_lag_x, begT, endT)
system = platform.system().lower()
# 分开训练
if if_separate_y == True:
    i = 0
    for yi_ind, yi in y.iteritems():
        X_train, X_test, y_train, y_test = train_test_split(X, yi,
                                                            train_size=0.75, test_size=0.25,
                                                            shuffle=False)
        cachedir = 'C:\\Downloads\\tpot_cache' if system == 'windows' else 'Users/Gary/Documents/tpot_cache'
        memory = Memory(location=cachedir, verbose=0)
        pipeline_optimizer = TPOTRegressor(generations=generations, population_size=population_size, cv=5,
                                           # TODO: 这里都有什么方法呢？
                                           template='Selector-Transformer-Regressor',
                                           scoring='r2',
                                           memory=memory,
                                           random_state=1996, verbosity=2)
        pipeline_optimizer.fit(X_train, y_train)
        print(pipeline_optimizer.score(X_test, y_test))
        pipeline_optimizer.export('./tpot_gen/separate_pipeline%d.py' % i)
        i += 1
        rmtree(cachedir)
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.75, test_size=0.25,
                                                        shuffle=False)
    cachedir = 'C:\\Downloads\\tpot_cache' if system == 'windows' else 'Users/Gary/Documents/tpot_cache'
    memory = Memory(location=cachedir, verbose=0)
    pipeline_optimizer = TPOTRegressor(generations=generations, population_size=population_size, cv=5,
                                       # TODO: 这里都有什么方法呢？
                                       template='Selector-Transformer-Regressor',
                                       scoring='r2',
                                       memory=memory,
                                       random_state=1996, verbosity=2)
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_test, y_test))
    pipeline_optimizer.export('./tpot_gen/multioutput_pipeline.py')
    rmtree(cachedir)
