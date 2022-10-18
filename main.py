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
import evaluator as ev

# �ļ��������
PATH_ORI_DATA = r'C:\Users\lucid\Documents\����ʵϰ\����֮���϶���\data'
if_update = False  ## ԭʼ�����ļ��Ƿ��Ѿ�����
use_cache = False  ## Ԥ�����߼�/������� or �����pickle��Ҫ����ʱ����ΪFalse (ע�����õ����ݸ�ʽ�������ñ�������Ԥ�Ȿ�����顣)
version = 'clklead1_rf03_1017'

# Ԥ�������
if_cls = True
align_to = 'month'
use_lag_x = 15
use_sup_lead = 1  ## ��������ʱ�ӵȲ�����
begT = '2004-01'
endT = datetime.date.today()
asset_sel = []

# ѵ������
n_splits = 10  ## ����ѵ������
test_size = 12
pipe = 'cls'  ## 'benchmark', 'post_FE'(reg), 'cls'
force_train = False  ## ��Ϊÿ��ʱ���ɸѡ����������һ�������Ա�������get dump��Ϊ�˽�ʡʱ����Կ���False
model_name = 'rf03'  ## 'separate'(use topot gen) or specific model name, availables see pipes file

#############Ԥ����##############
X, y_ret = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, align_to, use_lag_x, use_sup_lead, begT, endT)
if asset_sel:
    y_ret = y_ret.iloc[:, asset_sel]

if if_cls:
    y_cls = utils.reg_to_class(y_ret, 3)
    y = y_cls
else:
    y = y_ret

#############ѵ��##############
tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
models_list = {}
# ԭʼ��Xy��Ƭ֮ǰҪdeepcopy���������Ī������۸�ԭʼ����
for train_index, test_index in tscv.split(X.copy(deep=True)):
    if X.index[len(train_index)] < pd.Period('2015-7'):
        continue
    else:
        X_train, y_train = X.copy(deep=True).iloc[train_index, :], y.copy(deep=True).iloc[train_index, :]
        print("\nTRAIN period:", str(X_train.index[0]), '->', str(X_train.index[-1]),
              "\nStart training.......................")

        models = utils_train.get_models_dump(X_train, y_train, pipe=pipe, version=version, force_train=force_train,
                                             model_name=model_name)
        models_list[str(X_train.index[-1])] = deepcopy(models)

#############���Ժ�����##############
evalor_list = []
for train_index, test_index in tscv.split(X.copy(deep=True)):
    if X.index[len(train_index)] < pd.Period('2015-7'):
        continue
    else:
        X_train, X_test = X.copy(deep=True).iloc[train_index, :], X.copy(deep=True).iloc[test_index, :]
        y_train, y_test = y.copy(deep=True).iloc[train_index, :], y.copy(deep=True).iloc[test_index, :]
        y_test_ret = y_ret.copy(deep=True).loc[y_test.index, :]
        print("\nTEST period:", str(X_test.index[0]), '->', str(X_test.index[-1]),
              "\nStart testing...........................")
        # ���Ӳ��Լ�����ʹ��FE���Խ���
        X_test_long = utils.add_2years_test(X_train, X_test)

        evalor = Evaluator(models_list[str(X_train.index[-1])], if_cls, X_test_long, y_test, y_test_ret, X_train,
                           y_train)
        evalor_list.append(deepcopy(evalor))
        print("Test period:", str(X_test.index[0]), '->', str(X_test.index[-1]), "���껯��������Ϊ:",
              str(evalor.excess_ann_ret))
        del evalor

exc_rets = [i.excess_ann_ret for i in evalor_list]
port_ws, bench_ws = [i.port_worth for i in evalor_list], [i.bench_worth for i in evalor_list]
scoress = [i.scores for i in evalor_list]
port_poss = [i.port_pos for i in evalor_list]

con_pws = ev.get_continue_worth(port_ws)
con_bws = ev.get_continue_worth(bench_ws)

a=ev.get_perfo_hist(con_pws, con_bws)

b=ev.get_perfo_stats(con_pws, con_bws, a)

c=ev.get_asset_stats(scoress, port_poss, y_ret)

# ѵ������û��ʱ
# tscv = TimeSeriesSplit(n_splits=10)
# eval_list = []
# for train_index, test_index in tscv.split(X):
#     if X.index[len(train_index)] < pd.Period('2014-1'):
#         continue
#     else:
#         X_train, X_test = X.copy(deep=True).iloc[train_index, :], X.copy(deep=True).iloc[test_index, :]
#         y_train, y_test = y.copy(deep=True).iloc[train_index, :], y.copy(deep=True).iloc[test_index, :]
#         y_test_ret = y_ret.copy(deep=True).loc[y_test.index, :]
#         # ���Ӳ��Լ�����ʹ��FE���Խ���
#         X_test_long = utils.add_2years_test(X_train, X_test)
#         print('------------------�ָ���--------------------')
#         print("TRAIN period:", str(X_train.index[0]), '->', str(X_train.index[-1]),
#               "\nTEST period:", str(X_test.index[0]), '->', str(X_test.index[-1]), "\nStart get dump......")
#
#         # ��Ϊÿ��ѵ��ʱ���ɸѡ����������һ�������Ա�������ѵ����False force_trainΪ�˿���debug
#         models = utils_train.get_models_dump(X_train, y_train, version='cls', force_train=False)
#
#         evaluator = Evaluator(models, if_cls, X_test_long, y_test, y_test_ret, X_train, y_train)
#         eval_list.append(evaluator)
#         print("Test period:", str(X_test.index[0]), '->', str(X_test.index[-1]), "���껯��������Ϊ:",
#               str(evaluator.excess_ann_ret))
#         print('------------------һһ��ѵ�����Խ���--------------------')
#         # port_position, port_return, bench_return, port_worth, bench_worth, excess_ann_ret = evaluator.initializer()
