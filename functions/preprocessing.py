# Function: 数据预处理，包括
# 数据读取，
# 数据标准化，
# 数据集划分，按照8：2随机划分训练集和测试集
# 数据集保存, 保存为.pkl数据类型



# Train indices: [38 45 53 21 30 15  0 46 19 34 24 17 11  5 12  9 27 37 50 55 51 42 44 41
#  52 31 54 10 13 20  4 29 28 14 56 25 49  7 16 40 22 18 36 32 23]
# Test indices: [ 6  1  8 48 33 47  2 43  3 26 35 39]


import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.label[idx], dtype=torch.float)

def load_full(path):
    """
    读取全光谱数据和标签
    """
    df_data = pd.read_excel(path,sheet_name='Sheet1',header=None)
    df_label = pd.read_excel(path,sheet_name='Sheet2',header=None)

    # 全光谱数据是sheet1的第1列到第801列
    data = df_data.iloc[:,1:801].values

    # Cu标签是sheet2的第0列
    cu_label = df_label.iloc[:,0].values.reshape(-1,1)
    # Zn标签是sheet2的第1列
    zn_label = df_label.iloc[:,1].values.reshape(-1,1)
    # Pb标签是sheet2的第2列
    pb_label = df_label.iloc[:,2].values.reshape(-1,1)
    # V标签是sheet2的第3列
    v_label = df_label.iloc[:,3].values.reshape(-1,1)
    return data,cu_label, zn_label, pb_label, v_label



if __name__ == '__main__':

    # 设置路径
    full_spectrum_path = '/Users/guoyichen/Library/CloudStorage/OneDrive-UniversityofGlasgow/XRF实验/XRF/data/57样本全光谱.xlsx'
    selected_spectrum_path = '/Users/guoyichen/Library/CloudStorage/OneDrive-UniversityofGlasgow/XRF实验/XRF/data/57样本特征峰_360.xlsx' # 特征峰即使筛选过后的特征

    # 生成随机数据划分索引

    random_flag = False
    if random_flag:
        indices = np.arange(57)                             
        train_indices, test_indices = train_test_split(indices, test_size=0.2)
    else:
        train_indices = np.array([43, 26,  8, 17,  6,  4, 40, 19, 36, 48, 37, 53, 15,  9, 16, 24, 33, 54, 52, 25, 11, 32, 50, 49, 29, 41,  1, 21,  2, 44, 39, 35, 23, 46, 10, 22, 18, 56, 20,  7, 42, 14, 28, 51, 38])
        test_indices = np.array([ 0,  5, 30, 13, 34, 55, 27, 31, 45, 12, 47,  3])

    print("Train indices:", train_indices)
    # 45训练样本划分顺序:[43 26  8 17  6  4 40 19 36 48 37 53 15  9 16 24 33 54 52 25 11 32 50 49 29 41  1 21  2 44 39 35 23 46 10 22 18 56 20  7 42 14 28 51 38]
    print("Test indices:", test_indices)
    # 12测试样本划分顺序:[ 0  5 30 13 34 55 27 31 45 12 47  3]

    # 读取全光谱数据
    full_data, full_cu_label, full_zn_label, full_pb_label, full_v_label = load_full(full_spectrum_path)

    # print(full_data.shape)
    # print(full_cu_label.shape)
    # print(full_zn_label.shape)
    # print(full_pb_label.shape)
    # print(full_v_label.shape)

    # 不做归一化和标准化处理的效果反而还不错
    # 全光谱数据归一化 貌似不进行归一化的效果更好 如果这样归一化的话目前在Cu Zn V的表现是最好的
    # min_max_scaler = MinMaxScaler()
    # full_data = min_max_scaler.fit_transform(full_data.T).T

    # 全光谱数据标准化化 貌似不进行归一化的效果更好
    # scaler = StandardScaler()
    # full_data = scaler.fit_transform(full_data.T).T

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=10)
    pca_data = pca.fit_transform(full_data)
    
    # Selection
    from sklearn.feature_selection import SelectKBest, f_regression
    k_best = SelectKBest(f_regression, k=10)
    selected_data = k_best.fit_transform(full_data, full_cu_label.ravel())


    # 使用索引划分数据集
    full_data_train, full_data_test = full_data[train_indices], full_data[test_indices]
    full_cu_label_train, full_cu_label_test = full_cu_label[train_indices], full_cu_label[test_indices]
    full_zn_label_train, full_zn_label_test = full_zn_label[train_indices], full_zn_label[test_indices]
    full_pb_label_train, full_pb_label_test = full_pb_label[train_indices], full_pb_label[test_indices]
    full_v_label_train, full_v_label_test = full_v_label[train_indices], full_v_label[test_indices]

    # 如果dataset/全光谱 文件夹为空则，保存数据集,如果不为空，替换掉原有数据集
    if not os.path.exists('./dataset/全光谱'):
        os.makedirs('./dataset/全光谱')
    
    # 删除原有所有数据集
    for filename in os.listdir('./dataset/全光谱'):
        file_path = os.path.join('./dataset/全光谱', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # 如果dataset/全光谱pca 文件夹为空则，保存数据集,如果不为空，替换掉原有数据集
    if not os.path.exists('./dataset/全光谱pca'):
        os.makedirs('./dataset/全光谱pca')
    
    # 删除原有所有数据集
    for filename in os.listdir('./dataset/全光谱pca'):
        file_path = os.path.join('./dataset/全光谱pca', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # 如果dataset/全光谱特征选择 文件夹为空则，保存数据集,如果不为空，替换掉原有数据集
    if not os.path.exists('./dataset/全光谱特征选择'):
        os.makedirs('./dataset/全光谱特征选择')
    
    # 删除原有所有数据集
    for filename in os.listdir('./dataset/全光谱特征选择'):
        file_path = os.path.join('./dataset/全光谱特征选择', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # 保存新的数据集为.npy文件
    np.save('./dataset/全光谱/full_data_train.npy', full_data_train)
    np.save('./dataset/全光谱/full_data_test.npy', full_data_test)

    np.save('./dataset/全光谱pca/pca_data.npy', pca_data)
    np.save('./dataset/全光谱特征选择/selected_data.npy', selected_data)


    # 保存新的数据集为.npy文件
    np.save('./dataset/全光谱/full_Cu_label_train.npy', full_cu_label_train)
    np.save('./dataset/全光谱/full_Cu_label_test.npy', full_cu_label_test)
    np.save('./dataset/全光谱/full_Zn_label_train.npy', full_zn_label_train)
    np.save('./dataset/全光谱/full_Zn_label_test.npy', full_zn_label_test)
    np.save('./dataset/全光谱/full_Pb_label_train.npy', full_pb_label_train)
    np.save('./dataset/全光谱/full_Pb_label_test.npy', full_pb_label_test)
    np.save('./dataset/全光谱/full_V_label_train.npy', full_v_label_train)
    np.save('./dataset/全光谱/full_V_label_test.npy', full_v_label_test)



