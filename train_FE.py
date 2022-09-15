# coding=gbk
import copy
import pickle, platform, datetime
from os.path import abspath
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import talib
import utils, pipe_preproc, pipe_FE


# Configuration
PATH_ORI_DATA = r'C:\Users\lucid\Documents\长江实习\课题之自上而下\data'
## 原始数据文件是否已经更新
if_update = False
## 预处理逻辑(参数)变更/缓存的pickle需要更新时，设为False
use_cache = True
## 预处理参数
align_to = 'month'
use_lag_x = 15
begT = '2004-01'
endT = datetime.date.today()
## 训练参数
### 对于y的multi-output，是否分开训练
separate_y = True
generations = None
population_size = 100
max_time_mins = 120
cachedir = 'C:\\Downloads\\tpot_cache' if platform.system().lower() == 'windows' else abspath('../../Documents/tpot_cache')

# use_lag_x不再从这里传入，而是传入pipeline
X, y = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, align_to, use_lag_x, begT, endT)


ppl = Pipeline([
    ('features', pipe_FE.union),
    ('fillna', SimpleImputer(strategy='mean')),
    ('scaler1', StandardScaler()),
    ('selector1', pipe_FE.select_40n),
    # 这步之前列名没了，不是df不能操作
    ('series_to_supervised', pipe_preproc.SeriesToSupervised(n_in=use_lag_x, n_out=1)),
    # ('selector2', select_20n),
])

i = 0  # 可作为循环训练起点
for yi_ind, yi in y.iloc[:, i:].iteritems():
    X_supervised, yi = ppl.fit_transform(copy.deepcopy(X), yi)
    X_selected = pipe_FE.select_20n.fit_transform(X_supervised, yi)
    pipe, X_test, y_test = utils.generate_1_pipe_auto(X_selected, yi, generations, population_size,
                                                      max_time_mins, cachedir,
                                                      pipe_num=i)
    preds = pipe.predict(X_test)
    print('第%d个资产的Pipe r2 score:' % i, r2_score(y_test, preds))
    i += 1



