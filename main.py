# coding=gbk

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle, datetime
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import pandas as pd
import utils, utils_eda
from evaluator import Evaluator

# Configuration
PATH_ORI_DATA = r'C:\Users\lucid\Documents\����ʵϰ\����֮���϶���\data'
## ԭʼ�����ļ��Ƿ��Ѿ�����
if_update = False
## Ԥ�����߼�(����)���/�����pickle��Ҫ����ʱ����ΪFalse
####һ��Ҫע�����õ����ݸ�ʽ�������ñ�������Ԥ�Ȿ�����顣
use_cache = True
if_cls = True
## Ԥ�������
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
        print('------------------�ָ���--------------------')
        print("TRAIN period:", str(X_train.index[0]), '->', str(X_train.index[-1]),
              "\nTEST period:", str(X_test.index[0]), '->', str(X_test.index[-1]), "\nStart......")
        # ���Ӳ��Լ�����ʹ��FE���Խ���
        X_test_long = utils.add_2years_test(X_train, X_test)
        # ��Ϊÿ��splitɸѡ����������һ�������Ա�������ѵ����False force_trainΪ�˿��ٵ���
        models = utils.get_models_dump(X_train, y_train, version='cls', force_train=False)

        evaluator = Evaluator(models, if_cls, X_test_long, y_test, y_ret, X_train, y_train)
        eval_list.append(evaluator)
        print("Test period:", str(X_test.index[0]), '->', str(X_test.index[-1]), "���껯��������Ϊ:", str(evaluator.excess_ann_ret))

        # port_position, port_return, bench_return, port_worth, bench_worth, excess_ann_ret = evaluator.initializer()








