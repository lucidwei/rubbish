# coding=gbk

# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle, datetime
from sklearn.model_selection import train_test_split
import utils_eda
import utils
from evaluator import Evaluator


# Configuration
PATH_ORI_DATA = r'C:\Users\lucid\Documents\长江实习\课题之自上而下\data'
## 原始数据文件是否已经更新
if_update = False
## 预处理逻辑(参数)变更/缓存的pickle需要更新时，设为False
####一定要注意利用的数据格式，避免用本月行情预测本月行情。
use_cache = False
## 预处理参数
align_to = 'month'
use_lag_x = 13
begT = '2004-01'
endT = datetime.date.today()

X, y = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, use_lag_x, align_to, begT, endT)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    train_size=0.8, test_size=0.2,
                                                    shuffle=False)
# 增加测试集长度使得FE得以进行
X_test_long = utils.add_2years_test(X_train, X_test)
models = utils.get_models_dump(X_train, y_train, version='post_FE')

evaluator = Evaluator(models, X_test_long, y_test, X_train, y_train)

port_position, port_return, bench_return = evaluator.initializer()
port_worth = evaluator.get_port_worth()
bench_worth = evaluator.get_bench_worth()

# 画图见notebook
port_position.plot()
port_worth.plot()
bench_worth.plot()

# excess_return = port_return - bench0_return
#
