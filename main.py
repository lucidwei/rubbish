# coding=gbk

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import pickle, datetime
import sk_pipe
import utils
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tpot import TPOTRegressor

# Configuration
PATH_ORI_DATA = r'C:\Users\lucid\Documents\����ʵϰ\����֮���϶���\data'
## ԭʼ�����ļ��Ƿ��Ѿ�����
if_update = False
## �Ƿ�ʹ�û��������
use_cache = False
align_to = 'month'
begT='2005-01'
endT=datetime.date.today()

if not use_cache:
    # ��ȡ�ṹ������ʵ����updateΪTrue���ȡwindԴ���ݣ�Ĭ��False��ȡ֮ǰ����Ľṹ�������ļ�
    s_data = utils.GetStructuralData(PATH_ORI_DATA, update=if_update)
    raw_x = s_data.structural_data['x'][begT:endT].copy()
    raw_y = s_data.structural_data['y'].iloc[::-1][begT:endT].copy()

    # EDA(��Jupyter�н���)
    # eda.iter_graphs()
    # pre_data = pipeline.Preprocess(s_data.structural_data, align_to='month')

    # Ԥ����
    info = pd.read_excel(r'.\data\to_month_table.xlsx', index_col=0,
                         engine="openpyxl") if align_to == 'month' else \
        pd.read_excel(r'.\data\to_week_table.xlsx', index_col=0, engine="openpyxl")
    pipe_preprocess = Pipeline(steps=[
        ('special_treatment', sk_pipe.SpecialTreatment(info)),
        ('data_alignment', sk_pipe.DataAlignment(align_to, info)),
        ('get_stationary', sk_pipe.GetStationary(info)),
        ('series_to_supervised', sk_pipe.SeriesToSupervised(n_in=1, n_out=1))
    ])

    X,y = pipe_preprocess.fit_transform(raw_x, raw_y)
    print('...Pre-processing finished\n')

    # EDA compare ����ǰ�������
    # eda.iter_compare_graphs()

    # ��������
    with open('debug/prepipe_data', 'wb') as f:
        pickle.dump((X,y), f)
    pass
else:
    # ��ȡ��������
    with open('debug/prepipe_data', 'rb') as f:
        (X,y) = pickle.load(f)

# eda.iter_graphs(X, eda.get_info(), 60)
# model = pipeline.MacroModeling(data_dic)

X_train, X_test, y_train, y_test = train_test_split(X, y.iloc[:,0],
                                                    train_size=0.75, test_size=0.25)

pipeline_optimizer = TPOTRegressor(generations=100, population_size=100, cv=5,
                                    random_state=1996, verbosity=2)
pipeline_optimizer.fit(X_train, y_train)
print(pipeline_optimizer.score(X_test, y_test))
pipeline_optimizer.export('./tpot_gen/trial_pipeline1.py')

