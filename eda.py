# coding=gbk
# lucid   当前系统用户
# 2022/8/23   当前系统日期
# 17:01   当前系统时间
# PyCharm   创建文件的IDE名称
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re


def get_info():
    return pd.read_csv(r'.\data\info_table.csv', index_col=0, parse_dates=True)

def get_ori_id(id):
    if ')' in id:
        id_ori = re.split(r"\)", id)[1]
        if '_' in id_ori:
            id_ori = re.split(r'_', id_ori)[0]
    else:
        id_ori = id
    return id_ori

def plot_id(df: pd.DataFrame, id: str, info):
    sr = df[id].copy()
    # 取有数据点
    sr = sr[sr.first_valid_index():sr.last_valid_index()]
    # 剔除无数据点
    mask = np.isfinite(sr)
    # index转换成可画图格式
    if not pd.api.types.is_datetime64_ns_dtype(sr.index.dtype):
        sr.index = sr.index.end_time
        mask.index = mask.index.end_time

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(sr.index[mask], sr[mask], 'bo-')
    fig.autofmt_xdate()
    # 显示指标名
    id_ori = get_ori_id(id)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('指标ID:' + id +
              '\n指标名称:' + info.loc[id_ori, '指标名称'] +
              '\n频率:' + str(info.loc[id_ori, '频率']) +
              '\n单位:' + str(info.loc[id_ori, '单位']))
    print('指标ID:' + id)
    plt.show()
    plt.clf()


# Notebook用，需传入x数据和包含指标信息的info
def iter_graphs(x_df: pd.DataFrame, info, col_num: int, n=30):
    print('画从第%d到第%d列的图' %(col_num,col_num+n))
    for col in x_df.columns[col_num:col_num+n]:
        plot_id(x_df, col, info)


# 将预处理前后的两张图画在一起比较，看是否desirable
def iter_compare_graphs(x_df, x_df_new, info, col_num: int):
    # 每次画10张图
    for col in x_df.columns[col_num - 10:col_num]:
        sr = x_df[col].copy()
        sr1 = x_df_new[col].to_timestamp(freq='M', how='E').copy()
        # 取有数据点
        sr = sr[sr.first_valid_index():sr.last_valid_index()]
        # 剔除原df无数据点。高亮月度df缺失位置
        mask = np.isfinite(sr)
        mask1 = ~np.isfinite(sr1)

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(sr.index[mask], sr[mask], 'b-')
        ax.scatter(sr.index[mask], sr[mask], c='b', marker='o', alpha=0.5)
        ax.plot(sr1, 'r--', alpha=0.9)
        for i in sr1.index[mask1].date:
            ax.axvline(i)
        fig.autofmt_xdate()
        # 显示指标名
        print('指标ID:' + col)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.title('指标ID:' + col +
                  '\n指标名称:' + info.loc[col, '指标名称'] +
                  '\n频率:' + str(info.loc[col, '频率']) +
                  '\n单位:' + str(info.loc[col, '单位']))
        plt.show()
        plt.clf()
