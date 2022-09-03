# coding=gbk
# lucid   ��ǰϵͳ�û�
# 2022/8/16   ��ǰϵͳ����
# 14:47   ��ǰϵͳʱ��
# PyCharm   �����ļ���IDE����
import pickle, os.path, platform
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from tpot import TPOTRegressor
from joblib import Memory
from shutil import rmtree
import pipe_preproc

# ���wind excel���ݸ���
# minor warning: wind���ص��Զ���ϳ�ָ��û��������ָ��ID�������м�С�ĸ����ص���������feature��ʧ��8��19�հ汾�������޴����⡣
class GetStructuralData:
    # ����洢���ݣ��ֵ����ͣ����������෽�����ܰ���1��ʽ����ȡֵ������꣩2�����ݺ����������data�ļ���
    def __init__(self, path_read, update=False):
        print('update:', update)
        print('os.path.exists(csv):', os.path.exists(r'data/data_x.csv'))
        self.path_read = path_read
        self.update = update
        if update and os.path.exists(r'data/data_x.csv'):
            self.raw_data = self.read_files()
            self.info = self.get_info(last_info_row=6)
            self.structural_data = self.get_sdata(last_info_row=6)
        else:
            self.info = pd.read_csv(r'data/info_table.csv', index_col=0, parse_dates=True)
            self.structural_data = {'x': pd.read_csv(r'data/data_x.csv', index_col=0, parse_dates=True),
                                    'y': pd.read_csv(r'data/data_y.csv', index_col=0, parse_dates=True)}

    def read_files(self):
        md_dic = pd.read_excel(self.path_read + r'\���ں��.xlsx', engine="openpyxl", sheet_name=None)
        mo = pd.read_excel(self.path_read + r'\������.xlsx', engine="openpyxl")
        da = pd.read_excel(self.path_read + r'\��Ƶ�����ʲ�.xlsx', engine="openpyxl")
        di = pd.read_excel(self.path_read + r'\��Ƶ����ָ��.xlsx', engine="openpyxl")
        dic = {'macro_domestic': md_dic, 'macro_overseas': mo, 'daily_assets': da, 'daily_indicators': di}
        return dic

    def get_info(self, last_info_row):
        # TODO: ���ȼ��� ָ�����&����ʱ�㣨��ʡ���� �����Ż�ʱ���룩��
        # ÿ��һ��ָ�꣬columns����ָ��ID��ָ�����ơ����ð���Ƶ�ʣ���λ����Դ�����ң�����ʱ�䡣
        def by_tablename(tbname: str):
            info_by_tb = self.raw_data[tbname][:last_info_row + 1].T
            info_by_tb.set_axis(info_by_tb.iloc[:, 1], axis='index', inplace=True)
            info_by_tb.set_axis(info_by_tb.iloc[0, :], axis='columns', inplace=True)
            info_by_tb.drop('ָ��ID', axis=0, inplace=True)
            info_by_tb.drop('ָ��ID', axis=1, inplace=True)
            return info_by_tb

        def by_dicname(dicname: str):
            sheets = self.raw_data[dicname].keys()
            info_by_dic = None
            for i in sheets:
                # TODO: ���ȼ��� �߶˲�����������ϰwrapper
                info_by_tb = self.raw_data[dicname][i][:last_info_row + 1].T
                info_by_tb.set_axis(info_by_tb.iloc[:, 1], axis='index', inplace=True)
                info_by_tb.set_axis(info_by_tb.iloc[0, :], axis='columns', inplace=True)
                info_by_tb.drop('ָ��ID', axis=0, inplace=True)
                info_by_tb.drop('ָ��ID', axis=1, inplace=True)
                info_by_tb.insert(1, 'ָ�����', dicname + i)
                info_by_dic = pd.concat([info_by_dic, info_by_tb])
            return info_by_dic

        info_md = by_dicname('macro_domestic')
        info_mo = by_tablename('macro_overseas')
        info_di = by_tablename('daily_indicators')
        info_da = by_tablename('daily_assets')
        info = pd.concat([info_md, info_mo, info_di, info_da])
        if not os.path.exists(r'data/info_table.csv') or self.update:
            info.to_csv(r'.\data\info_table.csv')
        return info

    def get_sdata(self, last_info_row):
        def by_tablename(tbname: str):
            data_by_tb = self.raw_data[tbname]
            col = data_by_tb.iloc[1, :]
            col[0] = '����'
            data_by_tb.set_axis(col, axis='columns', inplace=True)
            # �����������Ч��Ϣ
            data_by_tb = data_by_tb[last_info_row + 1:-2].copy()
            data_by_tb.loc[:, '����'] = pd.DatetimeIndex(data_by_tb['����'])
            data_by_tb.set_index('����', inplace=True)
            return data_by_tb

        def by_dicname(dicname: str):
            sheets = self.raw_data[dicname].keys()
            data_by_dic = None
            for i in sheets:
                # TODO: ���ȼ��� ��ϰwrapper
                data_by_tb = self.raw_data[dicname][i]
                col = data_by_tb.iloc[1, :]
                col[0] = '����'
                data_by_tb.set_axis(col, axis='columns', inplace=True)
                data_by_tb = data_by_tb[last_info_row + 1:-2].copy()
                # ���ں���resample
                data_by_tb.loc[:, '����'] = pd.DatetimeIndex(data_by_tb['����'])
                data_by_tb.set_index('����', inplace=True)
                data_by_dic = pd.concat([data_by_dic, data_by_tb], axis=1)
            return data_by_dic

        data_md = by_dicname('macro_domestic')
        data_mo = by_tablename('macro_overseas')
        data_di = by_tablename('daily_indicators')
        sdata_y = by_tablename('daily_assets')
        sdata_x = pd.concat([data_md, data_mo, data_di], axis=1)
        if not os.path.exists(r'data/data_x.csv') or self.update:
            sdata_x.to_csv(r'.\data\data_x.csv')
            sdata_y.to_csv(r'.\data\data_y.csv')
        return {'x': sdata_x, 'y': sdata_y}


def get_preproc_data(ori_data_path, if_update, use_cache, align_to, use_lag_x, begT, endT):
    if not use_cache:
        # ��ȡ�ṹ������ʵ����updateΪTrue���ȡwindԴ���ݣ�Ĭ��False��ȡ֮ǰ����Ľṹ�������ļ�
        s_data = GetStructuralData(ori_data_path, update=if_update)
        raw_x = s_data.structural_data['x'][begT:endT].copy()
        raw_y = s_data.structural_data['y'].iloc[::-1][begT:endT].copy()

        # Ԥ����
        info = pd.read_excel(r'.\data\to_month_table.xlsx', index_col=0,
                             engine="openpyxl") if align_to == 'month' else \
            pd.read_excel(r'.\data\to_week_table.xlsx', index_col=0, engine="openpyxl")
        pipe_preprocess = Pipeline(steps=[
            ('special_treatment', pipe_preproc.SpecialTreatment(info)),
            ('data_alignment', pipe_preproc.DataAlignment(align_to, info)),
            ('get_stationary', pipe_preproc.GetStationary(info)),
            ('series_to_supervised', pipe_preproc.SeriesToSupervised(n_in=use_lag_x, n_out=1))
        ])

        X, y = pipe_preprocess.fit_transform(raw_x, raw_y)
        print('...Pre-processing finished\n')

        # ��������
        with open('debug/prepipe_data', 'wb') as f:
            pickle.dump((X, y), f)
        print('pickle saved')
    else:
        # ��ȡ��������
        with open('debug/prepipe_data', 'rb') as f:
            (X, y) = pickle.load(f)
        print('pickle loaded')

    return X, y


system = platform.system().lower()


def generate_1_pipe(X, y, generations, population_size, pipe_num=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.75, test_size=0.25,
                                                        shuffle=False)
    cachedir = 'C:\\Downloads\\tpot_cache' if system == 'windows' else 'Users/Gary/Documents/tpot_cache'
    memory = Memory(location=cachedir, verbose=0)
    pipeline_optimizer = TPOTRegressor(generations=generations, population_size=population_size, cv=5,
                                       # TODO: ���ﶼ��ʲô�����أ�
                                       template='Selector-Transformer-Regressor',
                                       scoring='r2',
                                       memory=memory,
                                       random_state=1996, verbosity=2)
    pipeline_optimizer.fit(X_train, y_train)
    print(pipeline_optimizer.score(X_test, y_test))
    if pipe_num is None:
        pipeline_optimizer.export('./tpot_gen/multioutput_tpotpipe.py')
    else:
        pipeline_optimizer.export('./tpot_gen/separate_tpotpipe%d.py' % pipe_num)
    rmtree(cachedir)
