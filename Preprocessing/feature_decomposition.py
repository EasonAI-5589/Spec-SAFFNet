import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from load import load_full

if __name__ == '__main__':
    ## 读取全光谱数据
    full_spectrum_path = './data/57样本全光谱.xlsx'

    full_data, full_cu_label, full_zn_label, full_pb_label, full_v_label = load_full(full_spectrum_path)

    ## 使用PCA进行特征分解
    pca = PCA(n_components=20)
    full_data = pca.fit_transform(full_data)
    print(full_data.shape)
    
    ## 数据归一化
    min_max_scaler = MinMaxScaler()
    full_data = min_max_scaler.fit_transform(full_data)

    ## 数据划分
    # 确定数据划分的索引
    train_indices = np.array([43, 26,  8, 17,  6,  4, 40, 19, 36, 48, 37, 53, 15,  9, 16, 24, 33, 54, 52, 25, 11, 32, 50, 49, 29, 41,  1, 21,  2, 44, 39, 35, 23, 46, 10, 22, 18, 56, 20,  7, 42, 14, 28, 51, 38])
    test_indices = np.array([ 0,  5, 30, 13, 34, 55, 27, 31, 45, 12, 47,  3])

    # 使用索引划分数据集
    full_data_train, full_data_test = full_data[train_indices], full_data[test_indices]
    full_cu_label_train, full_cu_label_test = full_cu_label[train_indices], full_cu_label[test_indices]
    full_zn_label_train, full_zn_label_test = full_zn_label[train_indices], full_zn_label[test_indices]
    full_pb_label_train, full_pb_label_test = full_pb_label[train_indices], full_pb_label[test_indices]
    full_v_label_train, full_v_label_test = full_v_label[train_indices], full_v_label[test_indices]

    ## 保存数据集
    # 设置保存路径
    save_path = './dataset/特征降维'

    # 如果dataset/全光谱 文件夹为空则，保存数据集,如果不为空，替换掉原有数据集
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 删除原有所有数据集
    for filename in os.listdir(save_path):
        file_path = os.path.join(save_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # 保存新的数据集为.npy文件
    np.save(save_path + '/full_data_train.npy', full_data_train)
    np.save(save_path + '/full_data_test.npy', full_data_test)