import os

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,f_regression

from load import load_full


if __name__ == '__main__':
    ## 读取全光谱数据
    full_spectrum_path = './data/57样本全光谱.xlsx'

    full_data, full_cu_label, full_zn_label, full_pb_label, full_v_label = load_full(full_spectrum_path)

    ## 使用k-best方法进行特征选择
    k_best = SelectKBest(score_func=f_regression, k=50)
    data_cu = k_best.fit_transform(full_data, full_cu_label.ravel())
    data_zn = k_best.fit_transform(full_data, full_zn_label.ravel())
    data_pb = k_best.fit_transform(full_data, full_pb_label.ravel())
    data_v = k_best.fit_transform(full_data, full_v_label.ravel())


    ## 数据归一化
    scalar = MinMaxScaler()
    data_cu = scalar.fit_transform(data_cu)
    data_zn = scalar.fit_transform(data_zn)
    data_pb = scalar.fit_transform(data_pb)
    data_v = scalar.fit_transform(data_v)

    ## 数据划分
    # 确定数据划分的索引
    train_indices = np.array([43, 26,  8, 17,  6,  4, 40, 19, 36, 48, 37, 53, 15,  9, 16, 24, 33, 54, 52, 25, 11, 32, 50, 49, 29, 41,  1, 21,  2, 44, 39, 35, 23, 46, 10, 22, 18, 56, 20,  7, 42, 14, 28, 51, 38])
    test_indices = np.array([ 0,  5, 30, 13, 34, 55, 27, 31, 45, 12, 47,  3])

    # 使用索引划分特征选择过后的数据集
    data_cu_train, data_cu_test = data_cu[train_indices], data_cu[test_indices]
    data_zn_train, data_zn_test = data_zn[train_indices], data_zn[test_indices]
    data_pb_train, data_pb_test = data_pb[train_indices], data_pb[test_indices]
    data_v_train, data_v_test = data_v[train_indices], data_v[test_indices]

    ## 保存数据集
    # 设置保存路径
    save_path = './dataset/特征选择'

    # 如果文件夹不存在则创建文件夹
    if not os.path.exists(save_path): 
        os.makedirs(save_path)

    # 删除原有数据集
    for filename in os.listdir(save_path): 
        file_path = os.path.join(save_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # 保存新的数据集为.npy文件
    np.save(save_path + '/data_cu_train.npy', data_cu_train)
    np.save(save_path + '/data_cu_test.npy', data_cu_test)
    np.save(save_path + '/data_zn_train.npy', data_zn_train)
    np.save(save_path + '/data_zn_test.npy', data_zn_test)
    np.save(save_path + '/data_pb_train.npy', data_pb_train)
    np.save(save_path + '/data_pb_test.npy', data_pb_test)
    np.save(save_path + '/data_v_train.npy', data_v_train)
    np.save(save_path + '/data_v_test.npy', data_v_test)
    
