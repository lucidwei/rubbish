# coding=gbk
import copy, datetime, pickle
from sklearn.metrics import r2_score, accuracy_score
import utils, pipe_pre_estimator, utils_train


# 文件处理参数
PATH_ORI_DATA = r'C:\Users\lucid\Documents\长江实习\课题之自上而下\data'
if_update = False  ## 原始数据文件是否已经更新
use_cache = True  ## 预处理逻辑/参数变更 or 缓存的pickle需要更新时，设为False (注意利用的数据格式，避免用本月行情预测本月行情。)

# 预处理参数
if_cls = True
align_to = 'month'
use_lag_x = 15
use_sup = True  ## 纳入美林时钟等补充框架
begT = '2004-01'
endT = datetime.date.today()
asset_sel = []

## 预训练参数
generations = None
population_size = 100
max_time_mins = 120
cachedir = utils.get_path('cachedir')

pipe = 'cls'  ## 'benchmark', 'post_FE'(reg), 'cls'
model_name = 'gbori'  ## 'separate'(use topot gen) or specific model name, availables see pipes file


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
    # i = 0  # 可作为循环训练起点
    # for yi_ind, yi in y.iloc[:, i:].iteritems():
    #     if if_cls:
    #         X_selected = pipe_pre_estimator.FE_ppl_cls.fit_transform(copy.deepcopy(X), yi)
    #     pipe, X_test, y_test = utils_train.generate_1_pipe_auto(if_cls, X_selected, yi, generations, population_size,
    #                                                       max_time_mins, cachedir,
    #                                                       pipe_num=i)
    #     preds = pipe.predict(X_test)
    #     # print('第%d个资产的Pipe r2 score:' % i, r2_score(y_test, preds))
    #     print('第%d个资产的Pipe accuracy_score:' % i, accuracy_score(y_test, preds))
    #     i += 1

    #####GSCV得到最佳hyper params
    param_grid = {
        # 'pipeline__gradientboostingclassifier__min_samples_split': [0.3, 0.5, 0.6],
        # 'pipeline__gradientboostingclassifier__max_features': [0.3, 1, 'auto'],
        'pipeline__gradientboostingclassifier__max_depth': [2],
        'pipeline__gradientboostingclassifier__min_samples_leaf': [0.3],
        #     'pipeline__gradientboostingclassifier__learning_rate': [0.01, 0.1, 0.5],
        #     'pipeline__gradientboostingclassifier__n_estimators': [100, 200],
        #     'pipeline__gradientboostingclassifier__subsample': [0.9, 1]
    }
    res_list = utils_train.get_gscv_result(X, y, pipe, model_name, param_grid)
    # 在一个资产里7分钟一个参数，要ignore warning
    # 写入缓存
    with open(r'models_dump/gscv/res_list', 'wb') as f:
        pickle.dump(res_list, f)
    print('res_list pickle saved')


