# coding=gbk

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle, datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import pandas as pd
import utils, utils_eda
from evaluator import Evaluator

# Configuration
PATH_ORI_DATA = r'C:\Users\lucid\Documents\长江实习\课题之自上而下\data'
## 原始数据文件是否已经更新
if_update = False
## 预处理逻辑(参数)变更/缓存的pickle需要更新时，设为False
####一定要注意利用的数据格式，避免用本月行情预测本月行情。
use_cache = True
if_cls = True
## 预处理参数
align_to = 'month'
use_lag_x = 13
begT = '2004-01'
endT = datetime.date.today()

X, y_ret = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, use_lag_x, align_to, begT, endT)
if if_cls:
    y_cls = utils.reg_to_class(y_ret, 3)
    y = y_cls
else:
    y = y_ret

tscv = TimeSeriesSplit(n_splits=5)
eval_list = []
for train_index, test_index in tscv.split(X):
    if X.index[len(train_index)] < pd.Period('2014-1'):
        continue
    else:
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]
        print('------------------分割线--------------------')
        print("TRAIN period:", str(X_train.index[0]), '->', str(X_train.index[-1]),
              "\nTEST period:", str(X_test.index[0]), '->', str(X_test.index[-1]), "\nStart......")
        # 增加测试集长度使得FE得以进行
        X_test_long = utils.add_2years_test(X_train, X_test)
        # 因为每个split筛选出的特征不一样，所以必须重新训练。False force_train为了快速迭代
        models = utils.get_models_dump(X_train, y_train, version='cls', force_train=False)

        evaluator = Evaluator(models, if_cls, X_test_long, y_test, y_ret, X_train, y_train)
        eval_list.append(evaluator)
        print("Test period:", str(X_test.index[0]), '->', str(X_test.index[-1]), "的年化超额收益为:", str(evaluator.excess_ann_ret))

        # port_position, port_return, bench_return, port_worth, bench_worth, excess_ann_ret = evaluator.initializer()








