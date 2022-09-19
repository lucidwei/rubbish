# coding=gbk
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score


# 该类中bench指等权组合，其他benchmark model需要将model list作为变量传入
class Evaluator:
    def __init__(self, models, X_test, y_test, X_train, y_train):
        self.models = models
        self.X_test = X_test
        self.y_test = y_test
        # 训练集净值、position需要train data
        self.X_train = X_train
        self.y_train = y_train
        self.port_pos = self.get_port_pos()
        self.port_ret = self.get_port_ret()
        self.bench_ret = self.get_bench_ret()

    def initializer(self):
        return self.port_pos, self.port_ret, self.bench_ret

    # 除了z-score还可以用percentile计算仓位
    def get_port_pos(self):
        # 对y_train求z-score时得到均值标准差，再针对pred和y_test normalize
        pos_info = pd.DataFrame(columns=['avg', 'std'], index=self.y_train.columns)
        for col_ind, col in self.y_train.iteritems():
            pos_info.loc[col_ind, 'avg'] = np.average(col)
            pos_info.loc[col_ind, 'std'] = np.std(col)
        # 将预测收益率转化为z-score
        pos_z = pd.DataFrame(index=self.y_test.index, columns=self.y_test.columns)
        i = 0
        for col_ind, col in self.y_test.iteritems():
            print('predicting test set for asset %d' % i)
            pred, predtrans_history = self.models[i].predict(self.X_test)
            print('第%d个资产的样本外 r2 score:' % i, r2_score(col, pred))
            pos_z.iloc[:, i] = (pred - pos_info.loc[col_ind, 'avg']) / pos_info.loc[col_ind, 'std']
            i += 1
        # z-score转化为position
        pos = pd.DataFrame(index=pos_z.index, columns=pos_z.columns)
        for row_ind, row in pos_z.iterrows():
            row[row < 0] = 0  # 不进行做空
            pos.loc[row_ind, :] = row / sum(row)

        return pos

    def get_port_ret(self):
        ret_df = pd.DataFrame(index=self.y_test.index, columns=['return'])
        for i in self.y_test.index:
            ret_df.loc[i, 'return'] = np.average(self.y_test.loc[i, :], weights=self.port_pos.loc[i, :])
        return ret_df

    # 等权组合
    def get_bench_ret(self):
        weights = [1 / len(self.y_test.columns) for _ in self.y_test.columns]
        # TODO: 利率债的return应该取相反数，简单起见忽略票息
        ret_df = pd.DataFrame(index=self.y_test.index, columns=['return'])
        for i in self.y_test.index:
            ret_df.loc[i, 'return'] = np.average(self.y_test.loc[i, :], weights=weights)
        # TODO: 检查这个和实际是否一致，怎么感觉数太小了
        return ret_df

    def get_port_worth(self):
        return return_to_worth(self.port_ret)

    def get_bench_worth(self):
        return return_to_worth(self.bench_ret)


# 月末净值
def return_to_worth(ret_df):
    worth_df = pd.DataFrame(index=ret_df.index, columns=['worth'])
    for i in range(len(worth_df.index)):
        if i == 0:
            worth_df.iloc[i, 0] = 1 + ret_df.iloc[i, 0]
        else:
            worth_df.iloc[i, 0] = (1 + ret_df.iloc[i, 0]) * worth_df.iloc[i - 1, 0]
    return worth_df
