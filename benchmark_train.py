# coding=gbk

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import datetime, platform
from os.path import abspath
from shutil import rmtree
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
population_size = 20
max_time_mins = 150

X, y = utils.get_preproc_data(PATH_ORI_DATA, if_update, use_cache, align_to, use_lag_x, begT, endT)

system = platform.system().lower()
cachedir = 'C:\\Downloads\\tpot_cache' if system == 'windows' else abspath('../../Documents/tpot_cache')
# �ֿ�ѵ��
if separate_y:
    i = 4 # ����Ϊѭ��ѵ�����
    for yi_ind, yi in y.iloc[:,i:].iteritems():
        utils.generate_1_pipe(X, yi, generations, population_size, max_time_mins, cachedir, i)
        i += 1
    rmtree(cachedir)
else:
    utils.generate_1_pipe(X, y, generations, population_size, max_time_mins, cachedir)
    rmtree(cachedir)
