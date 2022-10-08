# coding=gbk
# git config --global https.proxy http://127.0.0.1:7890
# git config --global --unset http.proxy
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle, datetime, copy
from copy import deepcopy
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import matplotlib.pyplot as plt
import pandas as pd
import utils, utils_eda, utils_train
from evaluator import Evaluator

# 文件处理参数
PATH_ORI_DATA = r'C:\Users\lucid\Documents\长江实习\课题之自上而下\data'
if_update = False  ## 原始数据文件是否已经更新
use_cache = True  ## 预处理逻辑/参数变更 or 缓存的pickle需要更新时，设为False (注意利用的数据格式，避免用本月行情预测本月行情。)
version = 'delcorr_1007'

# 预处理参数
if_cls = True
align_to = 'month'
use_lag_x = 14
use_sup = True  ## 纳入美林时钟等补充框架
begT = '2004-01'
endT = datetime.date.today()
asset_sel = [0, 2, 5, 7]

# 训练参数
n_splits = 5  ## 滚动训练次数
pipe = 'cls'  ## 'benchmark', 'post_FE'(reg), 'cls'
force_train = False  ## 因为每个时间段筛选出的特征不一样，所以必须重新get dump，为了节省时间调试可以False
model_name = 'rf'  ## 'separate'(use topot gen) or specific model name, availables see pipes file

#############预处理##############
X, y_ret = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, align_to, use_lag_x, use_sup, begT, endT)
if asset_sel:
    y_ret = y_ret.iloc[:, asset_sel]

if if_cls:
    y_cls = utils.reg_to_class(y_ret, 3)
    y = y_cls
else:
    y = y_ret

#############训练##############
tscv = TimeSeriesSplit(n_splits=n_splits)
models_list = {}
# 原始的Xy切片之前要deepcopy，否则可能莫名其妙篡改原始数据
for train_index, test_index in tscv.split(X.copy(deep=True)):
    if X.index[len(train_index)] < pd.Period('2014-1'):
        continue
    else:
        X_train, y_train = X.copy(deep=True).iloc[train_index, :], y.copy(deep=True).iloc[train_index, :]
        print("\nTRAIN period:", str(X_train.index[0]), '->', str(X_train.index[-1]),
              "\nStart training.......................")

        models = utils_train.get_models_dump(X_train, y_train, pipe=pipe, version=version, force_train=force_train,
                                             model_name=model_name)
        models_list[str(X_train.index[-1])] = deepcopy(models)

#############测试和评估##############
evalor_list = []
for train_index, test_index in tscv.split(X.copy(deep=True)):
    if X.index[len(train_index)] < pd.Period('2014-1'):
        continue
    else:
        X_train, X_test = X.copy(deep=True).iloc[train_index, :], X.copy(deep=True).iloc[test_index, :]
        y_train, y_test = y.copy(deep=True).iloc[train_index, :], y.copy(deep=True).iloc[test_index, :]
        y_test_ret = y_ret.copy(deep=True).loc[y_test.index, :]
        print("\nTEST period:", str(X_test.index[0]), '->', str(X_test.index[-1]),
              "\nStart testing...........................")
        # 增加测试集长度使得FE得以进行
        X_test_long = utils.add_2years_test(X_train, X_test)
        # debug feature names
        # names = models_list[str(X_train.index[-1])][0][:-1].get_feature_names_out()

        evalor = Evaluator(models_list[str(X_train.index[-1])], if_cls, X_test_long, y_test, y_test_ret, X_train,
                           y_train)
        evalor_list.append(deepcopy(evalor))
        print("Test period:", str(X_test.index[0]), '->', str(X_test.index[-1]), "的年化超额收益为:",
              str(evalor.excess_ann_ret))
        del evalor

exc_rets = [i.excess_ann_ret for i in evalor_list]
port_ws, bench_ws = [i.port_worth for i in evalor_list], [i.bench_worth for i in evalor_list]
scoress = [i.scores for i in evalor_list]
port_poss = [i.port_pos for i in evalor_list]



# 训练测试没拆开时
# tscv = TimeSeriesSplit(n_splits=10)
# eval_list = []
# for train_index, test_index in tscv.split(X):
#     if X.index[len(train_index)] < pd.Period('2014-1'):
#         continue
#     else:
#         X_train, X_test = X.copy(deep=True).iloc[train_index, :], X.copy(deep=True).iloc[test_index, :]
#         y_train, y_test = y.copy(deep=True).iloc[train_index, :], y.copy(deep=True).iloc[test_index, :]
#         y_test_ret = y_ret.copy(deep=True).loc[y_test.index, :]
#         # 增加测试集长度使得FE得以进行
#         X_test_long = utils.add_2years_test(X_train, X_test)
#         print('------------------分割线--------------------')
#         print("TRAIN period:", str(X_train.index[0]), '->', str(X_train.index[-1]),
#               "\nTEST period:", str(X_test.index[0]), '->', str(X_test.index[-1]), "\nStart get dump......")
#
#         # 因为每个训练时间段筛选出的特征不一样，所以必须重新训练。False force_train为了快速debug
#         models = utils_train.get_models_dump(X_train, y_train, version='cls', force_train=False)
#
#         evaluator = Evaluator(models, if_cls, X_test_long, y_test, y_test_ret, X_train, y_train)
#         eval_list.append(evaluator)
#         print("Test period:", str(X_test.index[0]), '->', str(X_test.index[-1]), "的年化超额收益为:",
#               str(evaluator.excess_ann_ret))
#         print('------------------一一轮训练测试结束--------------------')
#         # port_position, port_return, bench_return, port_worth, bench_worth, excess_ann_ret = evaluator.initializer()
