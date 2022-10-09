# coding=gbk
import copy, datetime
from sklearn.metrics import r2_score, accuracy_score
import utils, pipe_pre_estimator, utils_train


# �ļ��������
PATH_ORI_DATA = r'C:\Users\lucid\Documents\����ʵϰ\����֮���϶���\data'
if_update = False  ## ԭʼ�����ļ��Ƿ��Ѿ�����
use_cache = True  ## Ԥ�����߼�/������� or �����pickle��Ҫ����ʱ����ΪFalse (ע�����õ����ݸ�ʽ�������ñ�������Ԥ�Ȿ�����顣)

# Ԥ�������
if_cls = True
align_to = 'month'
use_lag_x = 15
use_sup = True  ## ��������ʱ�ӵȲ�����
begT = '2004-01'
endT = datetime.date.today()
asset_sel = []

## Ԥѵ������
generations = None
population_size = 100
max_time_mins = 120
cachedir = utils.get_path('cachedir')


#############Ԥ����##############
X, y_ret = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, align_to, use_lag_x, use_sup, begT, endT)
if asset_sel:
    y_ret = y_ret.iloc[:, asset_sel]

if if_cls:
    y_cls = utils.reg_to_class(y_ret, 3)
    y = y_cls
else:
    y = y_ret

if __name__ == "__main__":
    i = 0  # ����Ϊѭ��ѵ�����
    for yi_ind, yi in y.iloc[:, i:].iteritems():
        if if_cls:
            X_selected = pipe_pre_estimator.FE_ppl_cls.fit_transform(copy.deepcopy(X), yi)
        pipe, X_test, y_test = utils_train.generate_1_pipe_auto(if_cls, X_selected, yi, generations, population_size,
                                                          max_time_mins, cachedir,
                                                          pipe_num=i)
        preds = pipe.predict(X_test)
        # print('��%d���ʲ���Pipe r2 score:' % i, r2_score(y_test, preds))
        print('��%d���ʲ���Pipe accuracy_score:' % i, accuracy_score(y_test, preds))
        i += 1



