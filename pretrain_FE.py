# coding=gbk
import copy, datetime, pickle
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

pipe = 'cls'  ## 'benchmark', 'post_FE'(reg), 'cls'
model_name = 'logit'  ## 'separate'(use topot gen) or specific model name, availables see pipes file


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

    #####GSCV�õ����hyper params
    param_grid = {
        # 'pipeline__gradientboostingclassifier__min_samples_split': [0.5, 0.6, 0.7, 0.8], # 0.7���
        # 'pipeline__gradientboostingclassifier__max_features': [0.6, 0.7, 0.8, 0.9, 'sqrt'], # 'sqrt'asset1���ֺ�
        # 'pipeline__gradientboostingclassifier__max_depth': [2, 3],
        # 'pipeline__gradientboostingclassifier__min_samples_leaf': [0.3],
        # 'pipeline__gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.5],
        # 'pipeline__gradientboostingclassifier__n_estimators': [100, 200],
        # 'pipeline__gradientboostingclassifier__subsample': [0.9, 1]
        # 'pipeline__linearsvc__dual': [True, False],
        # 'pipeline__linearsvc__fit_intercept': [True, False],
        # 'pipeline__linearsvc__C': [0.01, 0.1, 1, 10],
        # 'pipeline__logisticregressioncv__fit_intercept': [True, False],
        # 'pipeline__logisticregressioncv__tol': [1e-5, 1e-4, 1e-3, 0.01]
        'pipeline__logisticregression__tol': [1e-4, 1e-3, 0.01],
        'pipeline__logisticregression__C': [1e-3, 0.01, 0.0464]
    }
    res_list = utils_train.get_gscv_result(X, y, pipe, model_name, param_grid)
    # ��һ���ʲ���7����һ��������Ҫignore warning
    # д�뻺��
    with open(r'models_dump/gscv/res_list', 'wb') as f:
        pickle.dump(res_list, f)
    print('res_list pickle saved')


