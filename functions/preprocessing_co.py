import os
import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle

class MyDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float), torch.tensor(self.label[idx], dtype=torch.float)

def load_selected(path):
    """
    读取专家筛选的数据和标签
    """
    df_co = pd.read_excel(path, sheet_name='Co', header=0)
    co_data = df_co.iloc[:, :-2].values
    co_label = df_co.iloc[:, -1:].values
    return co_data, co_label

# 设置路径
selected_spectrum_path = './data/selected_spectrum.xlsx'

# 读取专家筛选的Co数据
co_data, co_label = load_selected(selected_spectrum_path)

# 特征数据标准化
scalar = StandardScaler()
co_data = scalar.fit_transform(co_data)

# 随机划分数据集
co_data_train, co_data_test, co_label_train, co_label_test = train_test_split(co_data, co_label, test_size=0.2, random_state=42)

# 创建数据集
Co_train_Dataset = MyDataset(co_data_train, co_label_train)
Co_test_Dataset = MyDataset(co_data_test, co_label_test)

# 如果dataset文件夹为空则，保存数据集,如果不为空，替换掉原有数据集
if not os.path.exists('./dataset'):
    os.makedirs('./dataset')

# 删除原有Co数据集
for filename in os.listdir('./dataset'):
    if 'Co' in filename:
        file_path = os.path.join('./dataset', filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

# 保存新的Co数据集
with open('./dataset/Co_train_Dataset.pkl', 'wb') as f:
    pickle.dump(Co_train_Dataset, f)
with open('./dataset/Co_test_Dataset.pkl', 'wb') as f:
    pickle.dump(Co_test_Dataset, f)

print("Co数据集已重新划分并保存。")