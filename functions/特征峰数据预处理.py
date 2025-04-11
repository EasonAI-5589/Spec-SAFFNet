import os
import numpy as np
import pandas as pd

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


    # 设置特征峰路径
    selected_spectrum_path = '/Users/guoyichen/Library/CloudStorage/OneDrive-UniversityofGlasgow/XRF实验/XRF/data/57样本特征峰_360.xlsx'

    # 生成随机数据划分索引

    random_flag = False
    if random_flag:
        indices = np.arange(57)                             
        train_indices, test_indices = train_test_split(indices, test_size=0.2)
    else:
        train_indices = np.array([43, 26,  8, 17,  6,  4, 40, 19, 36, 48, 37, 53, 15,  9, 16, 24, 33, 54, 52, 25, 11, 32, 50, 49, 29, 41,  1, 21,  2, 44, 39, 35, 23, 46, 10, 22, 18, 56, 20,  7, 42, 14, 28, 51, 38])
        test_indices = np.array([ 0,  5, 30, 13, 34, 55, 27, 31, 45, 12, 47,  3])


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



    # 特征峰数据标准化 不用标准化的话数据的效果非常差
    scalar = StandardScaler()
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

    # # 创建数据集
    # Cu_train_Dataset = MyDataset(cu_data_train, cu_label_train)
    # Cu_test_Dataset = MyDataset(cu_data_test, cu_label_test)

    # Zn_train_Dataset = MyDataset(zn_data_train, zn_label_train)
    # Zn_test_Dataset = MyDataset(zn_data_test, zn_label_test)

    # Pb_train_Dataset = MyDataset(pb_data_train, pb_label_train)
    # Pb_test_Dataset = MyDataset(pb_data_test, pb_label_test)

    # V_train_Dataset = MyDataset(v_data_train, v_label_train)
    # V_test_Dataset = MyDataset(v_data_test, v_label_test)



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