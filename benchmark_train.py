# coding=gbk

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime
import utils

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
generations = 1
population_size = 2

X, y = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, align_to, use_lag_x, begT, endT)

# 分开训练
if separate_y:
    i = 0
    for yi_ind, yi in y.iteritems():
        utils.generate_1_pipe(X, yi, generations, population_size, i)
        i += 1
else:
    utils.generate_1_pipe(X, y, generations, population_size)
