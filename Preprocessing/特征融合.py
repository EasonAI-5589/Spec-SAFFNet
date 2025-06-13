import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from load import load_full, load_selected

# 特征峰
def peak():
    # 读取特征峰数据
    cu_data, cu_label, zn_data, zn_label, pb_data, pb_label, v_data, v_label = load_selected(selected_spectrum_path)
    # 特征峰数据标准化
    scalar = MinMaxScaler()
    cu_data = scalar.fit_transform(cu_data)
    zn_data = scalar.fit_transform(zn_data)
    pb_data = scalar.fit_transform(pb_data)
    v_data = scalar.fit_transform(v_data)

    return 


# 特征提取（PCA）
def extract_features_pca(full_data, n_components=20):
    """
    使用PCA进行特征提取
    :param full_data: 原始数据
    :param n_components: 保留的主成分数
    :return: 经过PCA处理后的数据
    """
    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(full_data)
    return transformed_data

# 特征选择（KBest）
def select_features_kbest(full_data, labels, k=50):
    """
    使用KBest进行特征选择
    :param full_data: 原始数据
    :param labels: 对应的标签
    :param k: 选择的特征数量
    :return: 选择后的数据
    """
    k_best = SelectKBest(score_func=f_regression, k=k)
    selected_data = k_best.fit_transform(full_data, labels.ravel())
    return selected_data

# 数据归一化
def normalize_data(data):
    """
    对数据进行归一化
    :param data: 输入数据
    :return: 归一化后的数据
    """
    scaler = MinMaxScaler()
    return scaler.fit_transform(data)

# 数据划分
def split_data(full_data, labels, train_indices, test_indices):
    """
    根据索引划分数据集
    :param full_data: 输入数据
    :param labels: 对应的标签
    :param train_indices: 训练集索引
    :param test_indices: 测试集索引
    :return: 训练集和测试集的划分结果
    """
    data_train = full_data[train_indices]
    data_test = full_data[test_indices]
    label_train = labels[train_indices]
    label_test = labels[test_indices]
    return data_train, data_test, label_train, label_test

# 数据保存
def save_data(save_path, data_dict):
    """
    保存处理后的数据
    :param save_path: 保存路径
    :param data_dict: 要保存的数据字典
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 删除原有数据集
    for filename in os.listdir(save_path):
        file_path = os.path.join(save_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # 保存新的数据集
    for filename, data in data_dict.items():
        np.save(os.path.join(save_path, filename), data)

# 主函数
if __name__ == '__main__':
    # 读取全光谱数据
    full_spectrum_path = './data/57样本全光谱.xlsx'
    full_data, full_cu_label, full_zn_label, full_pb_label, full_v_label = load_full(full_spectrum_path)

    # 读取特征峰数据
    selected_spectrum_path = './data/57样本特征峰_360.xlsx'


    # 特征提取（PCA）
    pca_data = extract_features_pca(full_data, n_components=20)
    
    # 特征选择（KBest）
    data_cu = select_features_kbest(full_data, full_cu_label, k=50)
    data_zn = select_features_kbest(full_data, full_zn_label, k=50)
    data_pb = select_features_kbest(full_data, full_pb_label, k=50)
    data_v = select_features_kbest(full_data, full_v_label, k=50)

    # 特征峰
    peak_cu,_,peak_zn,_,peak_pb,_,peak_v,_ = load_selected(selected_spectrum_path)

    # 数据归一化

    # PCA
    pca_data = normalize_data(pca_data)
    # K-Best
    data_cu = normalize_data(data_cu)
    data_zn = normalize_data(data_zn)
    data_pb = normalize_data(data_pb)
    data_v = normalize_data(data_v)
    # Peak
    peak_cu = normalize_data(peak_cu)
    peak_zn = normalize_data(peak_zn)
    peak_pb = normalize_data(peak_pb)
    peak_v = normalize_data(peak_v)

    # 数据划分（使用索引划分训练集和测试集）
    train_indices = np.array([43, 26, 8, 17, 6, 4, 40, 19, 36, 48, 37, 53, 15, 9, 16, 24, 33, 54, 52, 25, 11, 32, 50, 49, 29, 41, 1, 21, 2, 44, 39, 35, 23, 46, 10, 22, 18, 56, 20, 7, 42, 14, 28, 51, 38])
    test_indices = np.array([0, 5, 30, 13, 34, 55, 27, 31, 45, 12, 47, 3])

    # 将提取和选择后的数据进行融合（拼接）
    fused_data_cu = np.concatenate((pca_data, data_cu), axis=1)
    fused_data_zn = np.concatenate((pca_data, data_zn), axis=1)
    fused_data_pb = np.concatenate((pca_data, data_pb), axis=1)
    fused_data_v = np.concatenate((pca_data, data_v), axis=1)

    # 使用索引划分特征融合过后的数据集
    data_cu_train, data_cu_test = fused_data_cu[train_indices], fused_data_cu[test_indices]
    data_zn_train, data_zn_test = fused_data_zn[train_indices], fused_data_zn[test_indices]
    data_pb_train, data_pb_test = fused_data_pb[train_indices], fused_data_pb[test_indices]
    data_v_train, data_v_test = fused_data_v[train_indices], fused_data_v[test_indices]

    ## 保存数据集
    # 设置保存路径
    save_path = './dataset/融合特征'

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
    


    # # 划分数据集
    # fused_data_train, fused_data_test, _, _ = split_data(fused_data, None, train_indices, test_indices)
    # full_cu_label_train, full_cu_label_test = full_cu_label[train_indices], full_cu_label[test_indices]
    # full_zn_label_train, full_zn_label_test = full_zn_label[train_indices], full_zn_label[test_indices]
    # full_pb_label_train, full_pb_label_test = full_pb_label[train_indices], full_pb_label[test_indices]
    # full_v_label_train, full_v_label_test = full_v_label[train_indices], full_v_label[test_indices]

    # # 保存数据集
    # save_path = './dataset/融合数据'
    # data_to_save = {
    #     'fused_data_train.npy': fused_data_train,
    #     'fused_data_test.npy': fused_data_test,
    #     'full_cu_label_train.npy': full_cu_label_train,
    #     'full_cu_label_test.npy': full_cu_label_test,
    #     'full_zn_label_train.npy': full_zn_label_train,
    #     'full_zn_label_test.npy': full_zn_label_test,
    #     'full_pb_label_train.npy': full_pb_label_train,
    #     'full_pb_label_test.npy': full_pb_label_test,
    #     'full_v_label_train.npy': full_v_label_train,
    #     'full_v_label_test.npy': full_v_label_test
    # }

    # save_data(save_path, data_to_save)