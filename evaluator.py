# coding=gbk
import copy
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, accuracy_score


# 该类中bench指等权组合，其他benchmark model需要将model list作为变量传入
class Evaluator:
    def __init__(self, models, if_cls, X_test_long, y_test, y_ret, X_train, y_train):
        self.models = models
        self.if_cls = if_cls
        self.X_test_long = X_test_long
        self.y_test = y_test
        self.y_ret = y_ret
        # 训练集净值、position需要train data
        self.scores = {}
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

    # 除了z-score还可以用percentile计算仓位
    def get_port_pos(self):
        if not self.if_cls:
            # 对y_train求z-score时得到均值标准差，再针对pred和y_test normalize
            pos_info = pd.DataFrame(columns=['avg', 'std'], index=self.y_train.columns)
            for col_ind, col in self.y_train.iteritems():
                pos_info.loc[col_ind, 'avg'] = np.average(col)
                pos_info.loc[col_ind, 'std'] = np.std(col)
            # 将预测收益率转化为z-score
            pos_z = pd.DataFrame(index=self.y_test.index, columns=self.y_test.columns)
            i = 0
            for col_ind, real in self.y_test.iteritems():
                print('\npredicting test set for asset %d' % i)
                pred = self.models[i].predict(self.X_test_long)
                pred_short = pred[-len(real):]
                score = r2_score(real, pred_short)
                print('第%d个资产的样本外 r2 score:' % i, score)
                self.scores['第%d个资产:' % i] = score
                pos_z.iloc[:, i] = (pred_short - pos_info.loc[col_ind, 'avg']) / pos_info.loc[col_ind, 'std']
                i += 1
            # z-score转化为position
            pos = pd.DataFrame(index=pos_z.index, columns=pos_z.columns)
            for row_ind, row in pos_z.iterrows():
                row[row < 0] = 0  # 不进行做空
                pos.loc[row_ind, :] = row / sum(row)
        else:
            # 得到分类结果为2的位置矩阵
            pos_z = pd.DataFrame(index=self.y_test.index, columns=self.y_test.columns)
            i = 0
            for col_ind, real in self.y_test.iteritems():
                print('\npredicting test set for asset %d' % i)
                pred = self.models[i].predict(self.X_test_long)
                pred_short = pred[-len(real):]
                score = accuracy_score(real, pred_short)
                print('第%d个资产的样本外 accuracy score:' % i, score)
                self.scores['第%d个资产:' % i] = score
                pos_z.iloc[:, i] = [1 if j==2 else 0 for j in pred_short]
                i += 1
            # 对筛选出来的进行等权配置
            pos = pd.DataFrame(index=pos_z.index, columns=pos_z.columns)
            for row_ind, row in pos_z.iterrows():
                if sum(row) == 0:
                    pos.loc[row_ind, :] = np.zeros_like(row)
                    continue
                pos.loc[row_ind, :] = row / sum(row)
        return pos

    def get_port_ret(self):
        y_ret = self.y_ret if self.if_cls else self.y_test
        ret_df = pd.DataFrame(index=y_ret.index, columns=['return'])
        for i in y_ret.index:
            if self.port_pos.loc[i, :].sum() == 0:
                ret_df.loc[i, 'return'] = 0
                continue
            ret_df.loc[i, 'return'] = np.average(y_ret.loc[i, :], weights=self.port_pos.loc[i, :])
        return ret_df

    # 等权组合
    def get_bench_ret(self):
        y_ret = self.y_ret if self.if_cls else self.y_test
        weights = [1 / len(y_ret.columns) for _ in y_ret.columns]
        # TODO: 利率债的return应该取相反数，简单起见忽略票息
        ret_df = pd.DataFrame(index=y_ret.index, columns=['return'])
        for i in y_ret.index:
            ret_df.loc[i, 'return'] = np.average(y_ret.loc[i, :], weights=weights)
        # TODO: 检查这个和实际是否一致，怎么感觉数太小了
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


# 月末净值
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

def save_output(version, cpwl, cbwl, score_list, port_pos_list, y_ret):
    import os
    if not os.path.exists('data/results/'):
        os.makedirs('data/results/')
    perfo_hist = get_perfo_hist(cpwl, cbwl)
    perfo_hist.to_excel('data/results/'+version+'.xlsx', sheet_name='perfo')
    perfo_stats = get_perfo_stats(cpwl, cbwl, perfo_hist)
    asset_stats = get_asset_stats(score_list, port_pos_list, y_ret)

    with pd.ExcelWriter('data/results/'+version+'.xlsx', mode='a', engine='openpyxl') as writer:
        perfo_stats.to_excel(writer, sheet_name="perfo_stats")
        asset_stats.to_excel(writer, sheet_name="asset_stats")

    return

def get_perfo_hist(cpwl, cbwl):
    '''
    :param cpwl: conti_port_worth_list
    :param cbwl: conti_bench_worth_list
    :return:
    '''
    port_worth_hist = pd.DataFrame()
    bench_worth_hist = pd.DataFrame()
    for i in cpwl:
        port_worth_hist = pd.concat([port_worth_hist, i])
    for i in cbwl:
        bench_worth_hist = pd.concat([bench_worth_hist, i])
    df_perfo_hist = pd.concat([port_worth_hist, bench_worth_hist], axis=1)

    return df_perfo_hist

def get_perfo_stats(cpwl, cbwl, perfo_hist):
    # 日期，策略统计，基准统计
    #  其实有点毛病，第一个月收益没算进去
    def max_drawdown(array):
        drawdowns = []
        max_so_far = array[0]
        for i in range(len(array)):
            if array[i] > max_so_far:
                drawdown = 0
                drawdowns.append(drawdown)
                max_so_far = array[i]
            else:
                drawdown = max_so_far - array[i]
                drawdowns.append(drawdown)
        return (max(drawdowns))
    def sharpe_ratio(worth):
        ret = (worth[-1] - worth[0])/worth[0]
        sigma = worth.std()*np.sqrt(len(worth))
        return (ret/sigma)
    def ann_ret(worth):
        return ((worth[-1] - worth[0])/worth[0])/(len(worth)/12)

    index = [i.index[-1] for i in cpwl]
    df_perfo_stats = pd.DataFrame(index=index,
                                  columns=['port_年化收益率', 'port_夏普比率', 'port_最大回撤', '年化超额',
                                            '年化收益率', '夏普比率', '最大回撤'])
    for i in cbwl:
        df_perfo_stats.loc[i.index[-1], '年化收益率'] = ann_ret(np.array(i))
        df_perfo_stats.loc[i.index[-1], '夏普比率'] = sharpe_ratio(np.array(i))
        df_perfo_stats.loc[i.index[-1], '最大回撤'] = max_drawdown(np.array(i))
    for i in cpwl:
        df_perfo_stats.loc[i.index[-1], 'port_年化收益率'] = ann_ret(np.array(i))
        df_perfo_stats.loc[i.index[-1], 'port_夏普比率'] = sharpe_ratio(np.array(i))
        df_perfo_stats.loc[i.index[-1], 'port_最大回撤'] = max_drawdown(np.array(i))
        df_perfo_stats.loc[i.index[-1], '年化超额'] = ann_ret(np.array(i)) - df_perfo_stats.loc[i.index[-1], '年化收益率']
    overall = {
        'port_年化收益率':ann_ret(np.array(perfo_hist['port_worth'])), 'port_夏普比率':sharpe_ratio(np.array(perfo_hist['port_worth'])),
                  'port_最大回撤':max_drawdown(np.array(perfo_hist['port_worth'])), '年化超额':ann_ret(np.array(perfo_hist['port_worth'])) - ann_ret(np.array(perfo_hist['bench_worth'])),
        '年化收益率':ann_ret(np.array(perfo_hist['bench_worth'])), '夏普比率':sharpe_ratio(np.array(perfo_hist['bench_worth'])), '最大回撤':max_drawdown(np.array(perfo_hist['bench_worth']))
    }

    row_overall = pd.DataFrame(overall, columns=df_perfo_stats.columns, index=['整体'])
    df_perfo_stats= pd.concat([df_perfo_stats, row_overall], axis=0)

    return pd.DataFrame(df_perfo_stats, dtype=float).round(3)

def get_asset_stats(score_list, port_pos_list, y_ret):
    # 品种，预测准确率，持仓月数/总月数，贡献收益
    from utils_eda import get_ori_id, get_info, get_ori_name
    asset_ids = [get_ori_id(y_ret.columns[asset_num]) for asset_num in range(0,10)]
    asset_names = [get_ori_name(asset_id, get_info()) for asset_id in asset_ids]
    df_asset_stats = pd.DataFrame(index=asset_names, columns=['预测准确率', '持仓月数占比', '贡献收益(单利)'])

    score_df = pd.DataFrame()
    for i in score_list:
        score_df = pd.concat([score_df, pd.Series(i)], axis=1)
    avg_list = []
    for id, row in score_df.iterrows():
        avg_list.append(row.mean())
    df_asset_stats['预测准确率'] = np.array(avg_list)

    conti_port_pos = pd.DataFrame()
    for i in port_pos_list:
        conti_port_pos = pd.concat([conti_port_pos, i])
    hold_periods = (conti_port_pos != 0).sum(axis=0)
    df_asset_stats['持仓月数占比'] = np.array(hold_periods)/len(conti_port_pos.index)

    contrib_df = conti_port_pos * y_ret.loc[conti_port_pos.index,:]
    contrib_array = np.array(contrib_df.apply(lambda x: x.sum(),axis=0))
    df_asset_stats['贡献收益(单利)'] = contrib_array

    return df_asset_stats