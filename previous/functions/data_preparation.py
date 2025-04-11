

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from sklearn.preprocessing import StandardScaler, MinMaxScaler


class ToTensor(Dataset):
    def __init__(self, data, label):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label = torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]


def title_reader(path):
    """
    读取专家数据中的sheet name,作为标题
    """
    sheet_names = pd.ExcelFile(path).sheet_names
    titles = []
    for i in range(0,len(sheet_names)):
        titles.append(sheet_names[i])
    return titles


def data_reader(path):
    """
    以np数组的形式返回专家数据中5个元素的data和label
    """
    sheet_names = pd.ExcelFile(path).sheet_names
    data_sheet = []
    data = []
    label = []
    for i in range(0,len(sheet_names)):
        data_sheet.append(pd.read_excel(path, sheet_name=sheet_names[i],header=0))

    for i in range(0,len(data_sheet)):
        # 提取数据与标签，转化为numpy数组
        data.append(data_sheet[i].iloc[0:,:-2].values)
        label.append(data_sheet[i].iloc[0:,-1].values)

    data = np.array(data,dtype=object)
    label = np.array(label,dtype=object)

    return data,label
        

def dataset_split(data,label,ratio=0.8):
    """
    将数据集按照比例随机划分为训练集和测试集
    """
    train_data = []
    train_label = []
    test_data = []
    test_label = []

    for i in range(0,len(data)):
        # 随机划分训练集和测试集
        index = np.random.permutation(len(data[i]))
        train_index = index[0:int(ratio*len(data[i]))]
        test_index = index[int(ratio*len(data[i])):]

        train_data.append(data[i][train_index])
        train_label.append(label[i][train_index])
        test_data.append(data[i][test_index])
        test_label.append(label[i][test_index])

    train_data = np.array(train_data,dtype=object)
    train_label = np.array(train_label,dtype=object)
    test_data = np.array(test_data,dtype=object)
    test_label = np.array(test_label,dtype=object)

    return train_data,train_label,test_data,test_label
    

def data_loader(data,label):
    """
    将数据和标签转化为tensor, 准备训练集和加载器
    """

    dataset = ToTensor(data,label)
    dataloader = DataLoader(dataset, shuffle=True) # 使用shuffle打乱顺序

    return dataset, dataloader


def preprocessing(data):
    """
    对于训练数据进行标准化和归一化
    """
    scaler = StandardScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)

    return data



def one_dim_data_reader(path):
    """
    以np数组的形式返回专家数据中5个元素的data和label
    """
    sheet_names = pd.ExcelFile(path).sheet_names

    data = []
    label = []

    data_sheet = pd.read_excel(path, sheet_name=sheet_names[0],header=None)
    label_sheet = pd.read_excel(path, sheet_name=sheet_names[1],header=None)
    

    data.append(data_sheet.iloc[:,:].values)
    label.append(label_sheet.iloc[:,:].values)


    data = np.array(data)
    label = np.array(label)

    data = data.reshape((55, 2048))
    label = label.reshape(55,)


    my_data = [data,[],[],[],[]]
    my_label = [label,[],[],[],[]]    
    return my_data,my_label