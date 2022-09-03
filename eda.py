# coding=gbk
# lucid   ��ǰϵͳ�û�
# 2022/8/23   ��ǰϵͳ����
# 17:01   ��ǰϵͳʱ��
# PyCharm   �����ļ���IDE����
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
    # ȡ�����ݵ�
    sr = sr[sr.first_valid_index():sr.last_valid_index()]
    # �޳������ݵ�
    mask = np.isfinite(sr)
    # indexת���ɿɻ�ͼ��ʽ
    if not pd.api.types.is_datetime64_ns_dtype(sr.index.dtype):
        sr.index = sr.index.end_time
        mask.index = mask.index.end_time

    fig, ax = plt.subplots(figsize=(15, 8))
    ax.plot(sr.index[mask], sr[mask], 'bo-')
    fig.autofmt_xdate()
    # ��ʾָ����
    id_ori = get_ori_id(id)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.title('ָ��ID:' + id +
              '\nָ������:' + info.loc[id_ori, 'ָ������'] +
              '\nƵ��:' + str(info.loc[id_ori, 'Ƶ��']) +
              '\n��λ:' + str(info.loc[id_ori, '��λ']))
    print('ָ��ID:' + id)
    plt.show()
    plt.clf()


# Notebook�ã��贫��x���ݺͰ���ָ����Ϣ��info
def iter_graphs(x_df: pd.DataFrame, info, col_num: int, n=30):
    print('���ӵ�%d����%d�е�ͼ' %(col_num,col_num+n))
    for col in x_df.columns[col_num:col_num+n]:
        plot_id(x_df, col, info)


# ��Ԥ����ǰ�������ͼ����һ��Ƚϣ����Ƿ�desirable
def iter_compare_graphs(x_df, x_df_new, info, col_num: int):
    # ÿ�λ�10��ͼ
    for col in x_df.columns[col_num - 10:col_num]:
        sr = x_df[col].copy()
        sr1 = x_df_new[col].to_timestamp(freq='M', how='E').copy()
        # ȡ�����ݵ�
        sr = sr[sr.first_valid_index():sr.last_valid_index()]
        # �޳�ԭdf�����ݵ㡣�����¶�dfȱʧλ��
        mask = np.isfinite(sr)
        mask1 = ~np.isfinite(sr1)

        fig, ax = plt.subplots(figsize=(15, 8))
        ax.plot(sr.index[mask], sr[mask], 'b-')
        ax.scatter(sr.index[mask], sr[mask], c='b', marker='o', alpha=0.5)
        ax.plot(sr1, 'r--', alpha=0.9)
        for i in sr1.index[mask1].date:
            ax.axvline(i)
        fig.autofmt_xdate()
        # ��ʾָ����
        print('ָ��ID:' + col)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.title('ָ��ID:' + col +
                  '\nָ������:' + info.loc[col, 'ָ������'] +
                  '\nƵ��:' + str(info.loc[col, 'Ƶ��']) +
                  '\n��λ:' + str(info.loc[col, '��λ']))
        plt.show()
        plt.clf()
