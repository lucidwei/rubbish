# coding=gbk
import copy
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score


# ������benchָ��Ȩ��ϣ�����benchmark model��Ҫ��model list��Ϊ��������
class Evaluator:
    def __init__(self, models, if_cls, X_test_long, y_test, X_train, y_train):
        self.models = models
        self.if_cls = if_cls
        self.X_test_long = X_test_long
        self.y_test = y_test
        # ѵ������ֵ��position��Ҫtrain data
        self.X_train = X_train
        self.y_train = y_train
        self.port_pos = self.get_port_pos()
        self.port_ret = self.get_port_ret()
        self.bench_ret = self.get_bench_ret()
        self.port_worth = self.get_port_worth()
        self.bench_worth = self.get_bench_worth()
        self.excess_ann_ret = self.get_excess_ann_ret()

    def initializer(self):
        return self.port_pos, self.port_ret, self.bench_ret, self.port_worth, self.bench_worth, self.excess_ann_ret

    # ����z-score��������percentile�����λ
    def get_port_pos(self):
        if not self.if_cls:
            # ��y_train��z-scoreʱ�õ���ֵ��׼������pred��y_test normalize
            pos_info = pd.DataFrame(columns=['avg', 'std'], index=self.y_train.columns)
            for col_ind, col in self.y_train.iteritems():
                pos_info.loc[col_ind, 'avg'] = np.average(col)
                pos_info.loc[col_ind, 'std'] = np.std(col)
            # ��Ԥ��������ת��Ϊz-score
            pos_z = pd.DataFrame(index=self.y_test.index, columns=self.y_test.columns)
            i = 0
            for col_ind, real in self.y_test.iteritems():
                print('predicting test set for asset %d' % i)
                pred = self.models[i].predict(self.X_test_long)
                pred_short = pred[-len(real):]
                print('��%d���ʲ��������� r2 score:' % i, r2_score(real, pred_short))
                pos_z.iloc[:, i] = (pred_short - pos_info.loc[col_ind, 'avg']) / pos_info.loc[col_ind, 'std']
                i += 1
            # z-scoreת��Ϊposition
            pos = pd.DataFrame(index=pos_z.index, columns=pos_z.columns)
            for row_ind, row in pos_z.iterrows():
                row[row < 0] = 0  # ����������
                pos.loc[row_ind, :] = row / sum(row)
        else:
            # �õ�������Ϊ2��λ�þ���
            pos_z = pd.DataFrame(index=self.y_test.index, columns=self.y_test.columns)
            i = 0
            for col_ind, real in self.y_test.iteritems():
                print('predicting test set for asset %d' % i)
                pred = self.models[i].predict(self.X_test_long)
                pred_short = pred[-len(real):]
                print('��%d���ʲ��������� accuracy score:' % i, accuracy_score(real, pred_short))
                pos_z.iloc[:, i] = [1 if i==2 else 0 for i in pred_short]
                i += 1
            # ��Ȩ����
            pos = pd.DataFrame(index=pos_z.index, columns=pos_z.columns)
            for row_ind, row in pos_z.iterrows():
                pos.loc[row_ind, :] = row / sum(row)
        return pos

    def get_port_ret(self):
        ret_df = pd.DataFrame(index=self.y_test.index, columns=['return'])
        for i in self.y_test.index:
            ret_df.loc[i, 'return'] = np.average(self.y_test.loc[i, :], weights=self.port_pos.loc[i, :])
        return ret_df

    # ��Ȩ���
    def get_bench_ret(self):
        weights = [1 / len(self.y_test.columns) for _ in self.y_test.columns]
        # TODO: ����ծ��returnӦ��ȡ�෴�������������ƱϢ
        ret_df = pd.DataFrame(index=self.y_test.index, columns=['return'])
        for i in self.y_test.index:
            ret_df.loc[i, 'return'] = np.average(self.y_test.loc[i, :], weights=weights)
        # TODO: ��������ʵ���Ƿ�һ�£���ô�о���̫С��
        return ret_df

    def get_port_worth(self):
        pw = return_to_worth(self.port_ret)
        pw.columns = ['port_worth']
        return pw

    def get_bench_worth(self):
        bw = return_to_worth(self.bench_ret)
        bw.columns = ['bench_worth']
        return bw

    def get_excess_ann_ret(self):
        bench_annual = 12*self.bench_ret.mean()
        port_annual = 12*self.port_ret.mean()
        return port_annual-bench_annual


# ��ĩ��ֵ
def return_to_worth(ret_df):
    worth_df = pd.DataFrame(index=ret_df.index, columns=['worth'])
    for i in range(len(worth_df.index)):
        if i == 0:
            worth_df.iloc[i, 0] = 1 + ret_df.iloc[i, 0]
        else:
            worth_df.iloc[i, 0] = (1 + ret_df.iloc[i, 0]) * worth_df.iloc[i - 1, 0]
    return worth_df


def get_continue_worth(ws):
    ws = copy.deepcopy(ws)
    con = [ws[0]]
    for i in range(1, len(ws)):
        ws[i] = ws[i] * ws[i-1].iloc[-1, :]
        con.append(ws[i])
    return con