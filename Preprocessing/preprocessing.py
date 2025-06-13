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

def load_selected(path):
    """
    读取专家筛选的数据和标签
    """
    df_cu = pd.read_excel(path,sheet_name='Cu',header=0)
    df_zn = pd.read_excel(path,sheet_name='Zn',header=0)
    df_pb = pd.read_excel(path,sheet_name='Pb',header=0)
    df_v = pd.read_excel(path,sheet_name='V',header=0)

    cu_data = df_cu.iloc[:,:-2].values
    cu_label = df_cu.iloc[:,-1:].values

    zn_data = df_zn.iloc[:,:-2].values
    zn_label = df_zn.iloc[:,-1:].values

    pb_data = df_pb.iloc[:,:-2].values
    pb_label = df_pb.iloc[:,-1:].values

    v_data = df_v.iloc[:,:-2].values
    v_label = df_v.iloc[:,-1:].values


    return cu_data, cu_label, zn_data, zn_label, pb_data, pb_label, v_data, v_label


if __name__ == '__main__':

    # 设置路径
    full_spectrum_path = './data/57样本全光谱.xlsx'
    selected_spectrum_path = './data/57样本特征峰_360.xlsx' # 特征峰即使筛选过后的特征

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

    # 全光谱数据归一化
    min_max_scaler = MinMaxScaler()
    full_data = min_max_scaler.fit_transform(full_data)

    # 使用索引划分数据集
    full_data_train, full_data_test = full_data[train_indices], full_data[test_indices]
    full_cu_label_train, full_cu_label_test = full_cu_label[train_indices], full_cu_label[test_indices]
    full_zn_label_train, full_zn_label_test = full_zn_label[train_indices], full_zn_label[test_indices]
    full_pb_label_train, full_pb_label_test = full_pb_label[train_indices], full_pb_label[test_indices]
    full_v_label_train, full_v_label_test = full_v_label[train_indices], full_v_label[test_indices]

    # 创建数据集

    # 如果dataset/全光谱 文件夹为空则，保存数据集,如果不为空，替换掉原有数据集
    if not os.path.exists('./dataset/全光谱'):
        os.makedirs('./dataset/全光谱')
    
    # 删除原有所有数据集
    for filename in os.listdir('./dataset/全光谱'):
        file_path = os.path.join('./dataset/全光谱', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

    # 保存新的数据集为.npy文件
    np.save('./dataset/全光谱/full_data_train.npy', full_data_train)
    np.save('./dataset/全光谱/full_data_test.npy', full_data_test)
    np.save('./dataset/全光谱/full_Cu_label_train.npy', full_cu_label_train)
    np.save('./dataset/全光谱/full_Cu_label_test.npy', full_cu_label_test)
    np.save('./dataset/全光谱/full_Zn_label_train.npy', full_zn_label_train)
    np.save('./dataset/全光谱/full_Zn_label_test.npy', full_zn_label_test)
    np.save('./dataset/全光谱/full_Pb_label_train.npy', full_pb_label_train)
    np.save('./dataset/全光谱/full_Pb_label_test.npy', full_pb_label_test)
    np.save('./dataset/全光谱/full_V_label_train.npy', full_v_label_train)
    np.save('./dataset/全光谱/full_V_label_test.npy', full_v_label_test)

    # 读取特征峰数据
    cu_data, cu_label, zn_data, zn_label, pb_data, pb_label, v_data, v_label = load_selected(selected_spectrum_path)

    # print("-------------------")
    # print(cu_label.shape)
    # print(cu_data.shape)
    # print("-------------------")

    # print(zn_label.shape)
    # print(zn_data.shape)
    # print("-------------------")

    # print(pb_label.shape)
    # print(pb_data.shape)
    # print("-------------------")

    # print(v_label.shape)
    # print(v_data.shape)
    # print("-------------------")



    # 特征峰数据标准化
    scalar = MinMaxScaler()
    cu_data = scalar.fit_transform(cu_data)
    zn_data = scalar.fit_transform(zn_data)
    pb_data = scalar.fit_transform(pb_data)
    v_data = scalar.fit_transform(v_data)



    # 使用索引划分数据集
    cu_data_train, cu_data_test = cu_data[train_indices], cu_data[test_indices]
    cu_label_train, cu_label_test = cu_label[train_indices], cu_label[test_indices]

    zn_data_train, zn_data_test = zn_data[train_indices], zn_data[test_indices]
    zn_label_train, zn_label_test = zn_label[train_indices], zn_label[test_indices]

    pb_data_train, pb_data_test = pb_data[train_indices], pb_data[test_indices]
    pb_label_train, pb_label_test = pb_label[train_indices], pb_label[test_indices]

    v_data_train, v_data_test = v_data[train_indices], v_data[test_indices]
    v_label_train, v_label_test = v_label[train_indices], v_label[test_indices]

    # print("-------------------")
    # print(zn_label_test)
    # print("-------------------")
    # print(cu_label_test)
    # print("-------------------")
    # print(pb_label_test)
    # print("-------------------")


    # 如果dataset/特征峰 文件夹为空则，保存数据集,如果不为空，替换掉原有数据集
    if not os.path.exists('./dataset/特征峰'):
        os.makedirs('./dataset/特征峰')
    
    # 删除原有所有数据集
    for filename in os.listdir('./dataset/特征峰'):
        file_path = os.path.join('./dataset/特征峰', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
    
    # 保存新的数据集为.npy文件
    np.save('./dataset/特征峰/Cu_train_data.npy', cu_data_train)
    np.save('./dataset/特征峰/Cu_train_label.npy', cu_label_train)
    np.save('./dataset/特征峰/Cu_test_data.npy', cu_data_test)
    np.save('./dataset/特征峰/Cu_test_label.npy', cu_label_test)
    
    np.save('./dataset/特征峰/Zn_train_data.npy', zn_data_train)
    np.save('./dataset/特征峰/Zn_train_label.npy', zn_label_train)
    np.save('./dataset/特征峰/Zn_test_data.npy', zn_data_test)
    np.save('./dataset/特征峰/Zn_test_label.npy', zn_label_test)
    
    np.save('./dataset/特征峰/Pb_train_data.npy', pb_data_train)
    np.save('./dataset/特征峰/Pb_train_label.npy', pb_label_train)
    np.save('./dataset/特征峰/Pb_test_data.npy', pb_data_test)
    np.save('./dataset/特征峰/Pb_test_label.npy', pb_label_test)
    
    np.save('./dataset/特征峰/V_train_data.npy', v_data_train)
    np.save('./dataset/特征峰/V_train_label.npy', v_label_train)
    np.save('./dataset/特征峰/V_test_data.npy', v_data_test)
    np.save('./dataset/特征峰/V_test_label.npy', v_label_test)



    # # 比较cu_label_test和full_cu_label_test是否完全一致,返回的是一个True或者False
    # print(cu_label_test)
    # print(full_cu_label_test)
    # print(np.array_equal(cu_label_test, full_cu_label_test))