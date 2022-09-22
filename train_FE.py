# coding=gbk
import copy
import pickle, platform, datetime
from os.path import abspath
from sklearn.metrics import r2_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import talib
import utils, pipe_preproc, pipe_FE


# Configuration
PATH_ORI_DATA = r'C:\Users\lucid\Documents\����ʵϰ\����֮���϶���\data'
## ԭʼ�����ļ��Ƿ��Ѿ�����
if_update = False
## Ԥ�����߼�(����)���/�����pickle��Ҫ����ʱ����ΪFalse
use_cache = True
## Ԥ�������
align_to = 'month'
use_x_lags = 13
begT = '2004-01'
endT = datetime.date.today()
## ѵ������
### ����y��multi-output���Ƿ�ֿ�ѵ��
separate_y = True
if_cls = True
generations = None
population_size = 100
max_time_mins = 1
cachedir = utils.get_path('cachedir')


if __name__ == "__main__":
    X, y = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, use_x_lags, align_to, begT, endT)
    if if_cls:
        y_cls = utils.reg_to_class(y, 3)
        y = y_cls
    i = 0  # ����Ϊѭ��ѵ�����
    for yi_ind, yi in y.iloc[:, i:].iteritems():
        if if_cls:
            X_selected, yi = pipe_FE.FE_ppl_cls.fit_transform(copy.deepcopy(X), yi)
        pipe, X_test, y_test = utils.generate_1_pipe_auto(if_cls, X_selected, yi, generations, population_size,
                                                          max_time_mins, cachedir,
                                                          pipe_num=i)
        preds = pipe.predict(X_test)
        # print('��%d���ʲ���Pipe r2 score:' % i, r2_score(y_test, preds))
        print('��%d���ʲ���Pipe accuracy_score:' % i, accuracy_score(y_test, preds))
        i += 1



