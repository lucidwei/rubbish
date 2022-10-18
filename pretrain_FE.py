# coding=gbk
import copy, datetime, pickle
from sklearn.metrics import r2_score, accuracy_score
import utils, pipe_pre_estimator, utils_train


# 文件处理参数
PATH_ORI_DATA = r'C:\Users\lucid\Documents\长江实习\课题之自上而下\data'
if_update = False  ## 原始数据文件是否已经更新
use_cache = False  ## 预处理逻辑/参数变更 or 缓存的pickle需要更新时，设为False (注意利用的数据格式，避免用本月行情预测本月行情。)

# 预处理参数
if_cls = True
align_to = 'month'
use_lag_x = 15
use_sup = 4  ## 纳入美林时钟等补充框架
begT = '2004-01'
endT = datetime.date.today()
asset_sel = []

## 预训练参数
generations = None
population_size = 100
max_time_mins = 120
cachedir = utils.get_path('cachedir')

pipe = 'cls'  ## 'benchmark', 'reg_FE', 'cls'
model_name = 'sgdcv'  ## 'separate'(use topot gen) or specific model name, availables see pipes file


#############预处理##############
X, y_ret = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, align_to, use_lag_x, use_sup, begT, endT)
if asset_sel:
    y_ret = y_ret.iloc[:, asset_sel]

if if_cls:
    y_cls = utils.reg_to_class(y_ret, 3)
    y = y_cls
else:
    y = y_ret

if __name__ == "__main__":

    #####GSCV得到最佳hyper params
    param_grid = {
        # 'pipeline__gradientboostingclassifier__min_samples_split': [0.5, 0.6, 0.7, 0.8], # 0.7最好
        # 'pipeline__gradientboostingclassifier__max_features': [0.6, 0.7, 0.8, 0.9, 'sqrt'], # 'sqrt'asset1表现好
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
        # 'pipeline__logisticregression__tol': [1e-4, 1e-3, 0.01],
        # 'pipeline__logisticregression__C': [1e-3, 0.01, 0.0464],
        # 'pipeline-2__elasticnet__alpha': [1e-4, 1e-2, 1],
        # 'pipeline-2__elasticnet__l1_ratio': [0.5, 0.7, 0.9],
        # 'pipeline__randomforestclassifier__oob_score': [True, False],
        # 'pipeline__randomforestclassifier__max_depth': [2, 3],
        # 'pipeline__randomforestclassifier__min_samples_split': [0.1, 0.2, 0.3, 4, 6, 12, 18],
        'pipeline__sgdclassifier__alpha': [1e-2, 1e-3],
        'pipeline__sgdclassifier__eta0': [0.1, 0.01, 0.001],
        'pipeline__sgdclassifier__l1_ratio': [1, 0.75, 0.5]
    }
    res_list = utils_train.get_gscv_result(X, y, pipe, model_name, param_grid)
    # 在一个资产里7分钟一个参数，要ignore warning
    # 写入缓存
    with open(r'models_dump/gscv/res_list', 'wb') as f:
        pickle.dump(res_list, f)
    print('res_list pickle saved')


