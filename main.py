# coding=gbk

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle, datetime
from sklearn.model_selection import train_test_split
import utils_eda
import utils
from evaluator import Evaluator


# Configuration
PATH_ORI_DATA = r'C:\Users\lucid\Documents\����ʵϰ\����֮���϶���\data'
## ԭʼ�����ļ��Ƿ��Ѿ�����
if_update = False
## Ԥ�����߼�(����)���/�����pickle��Ҫ����ʱ����ΪFalse
####һ��Ҫע�����õ����ݸ�ʽ�������ñ�������Ԥ�Ȿ�����顣
use_cache = False
## Ԥ�������
align_to = 'month'
use_lag_x = 13
begT = '2004-01'
endT = datetime.date.today()

X, y = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, use_lag_x, align_to, begT, endT)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8, test_size=0.2,
                                                    shuffle=False)
# ���Ӳ��Լ�����ʹ��FE���Խ���
X_test_long = utils.add_2years_test(X_train, X_test)
models = utils.get_models_dump(X_train, y_train, version='post_FE')

evaluator = Evaluator(models, X_test_long, y_test, X_train, y_train)

port_position, port_return, bench_return = evaluator.initializer()
port_worth = evaluator.get_port_worth()
bench_worth = evaluator.get_bench_worth()

# ��ͼ��notebook
port_position.plot()
port_worth.plot()
bench_worth.plot()

# excess_return = port_return - bench0_return
#
