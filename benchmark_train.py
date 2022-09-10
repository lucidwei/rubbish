# coding=gbk

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime, platform
from os.path import abspath
from shutil import rmtree
from sklearn.metrics import r2_score
import utils

# Configuration
PATH_ORI_DATA = r'C:\Users\lucid\Documents\����ʵϰ\����֮���϶���\data'
## ԭʼ�����ļ��Ƿ��Ѿ�����
if_update = False
## Ԥ�����߼�(����)���/�����pickle��Ҫ����ʱ����ΪFalse
use_cache = True
## Ԥ�������
align_to = 'month'
use_lag_x = 15
begT = '2004-01'
endT = datetime.date.today()
## ѵ������
### ����y��multi-output���Ƿ�ֿ�ѵ��
separate_y = True
generations = None
population_size = 30
max_time_mins = 180
# tpot_config = 'TPOT MDR' //������

X, y = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, align_to, use_lag_x, begT, endT)

system = platform.system().lower()
cachedir = 'C:\\Downloads\\tpot_cache' if system == 'windows' else abspath('../../Documents/tpot_cache')

# tpotѵ��
if separate_y:
    i = 0  # ����Ϊѭ��ѵ�����
    for yi_ind, yi in y.iloc[:, i:].iteritems():
        pipe, X_test, y_test = utils.generate_1_pipe(X, yi, generations, population_size, max_time_mins, cachedir,
                                                     pipe_num=i, tpot_config=utils.tpot_config)
        preds = pipe.predict(X_test)
        print('��%d���ʲ���Pipe r2 score:' % i, r2_score(y_test, preds))
        i += 1
    # rmtree(cachedir)
else:
    utils.generate_1_pipe(X, y, generations, population_size, max_time_mins, cachedir)
    # rmtree(cachedir)

# tpot-generated modelsѵ��
models = utils.get_models_dumps()

# from sklearn.ensemble import ExtraTreesRegressor
# from sklearn.feature_selection import SelectPercentile, mutual_info_regression
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.model_selection import train_test_split
#
# exported_pipeline = make_pipeline(
#     SelectPercentile(score_func=mutual_info_regression, percentile=2),
#     PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
#     ExtraTreesRegressor(bootstrap=True, max_features=0.05, min_samples_leaf=16, min_samples_split=4, n_estimators=100)
# )
#
# X_train, X_test, y_train, y_test = train_test_split(X, y,
#                                                     train_size=0.8, test_size=0.2,
#                                                     shuffle=False)
# i=0
# exported_pipeline.fit(X_train, y_train.iloc[:, i])
# results = exported_pipeline.predict(X_test)
# print('��%d���ʲ���Pipe r2 score:' % i, r2_score(y_test.iloc[:, i], results))
# a=1