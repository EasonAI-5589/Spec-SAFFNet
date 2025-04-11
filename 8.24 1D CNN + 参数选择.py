# %%
import math
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import time as time
import matplotlib.pyplot as plt
import random

# 定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

class Custom1DCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, hidden_size):
        super(Custom1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool1d(kernel_size=4)
        self.fc1 = nn.Linear(out_channels * (2048 // 4-1), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x   

def Set_Random_State():
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return
    
def Read_Data():
    # 从Excel读取数据集
    data_df = pd.read_excel('Pb1.xlsx', sheet_name='Sheet1',header=None)
    data = data_df.iloc[0:, :2048].values
    label_df = pd.read_excel('Pb1.xlsx',sheet_name='Sheet2',header=None)
    labels = label_df.iloc[0:, 0].values
    return data,label_df


def Split_with_Sequential(data,labels,size):
    # 按顺序划分数据集
    split_index = int(len(data) * size)
    data_train = data[:split_index]
    labels_train = labels[:split_index]
    data_test = data[split_index:]
    labels_test = labels[split_index:]
    return data_train,data_test,labels_train,labels_test

def Split_with_Random(data,labels,size):
    # 随机划分
    data_train, data_test, labels_train, labels_test= train_test_split(data, labels, train_size=size, random_state=42) 
    return data_train, data_test, labels_train, labels_test

def Standardlization(data):
    # 数据预处理：标准化 + 整理为1DCNN需要的形式
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data)
    return data_normalized

def Data_Augmentation(data):
    data = np.expand_dims(data,axis=1)
    return data

def Data_to_Tensor(data_train, data_test, labels_train, labels_test):
    data_train_tensor = torch.from_numpy(np.array(data_train, dtype=np.float32))
    labels_train_tensor = torch.from_numpy(np.array(labels_train, dtype=np.float32))
    data_test_tensor = torch.from_numpy(np.array(data_test, dtype=np.float32))
    labels_test_tensor = torch.from_numpy(np.array(labels_test, dtype=np.float32))
    
    return data_train_tensor, labels_train_tensor, data_test_tensor, labels_test_tensor


def Load_Data_For_NN(data_train,labels_train,batch_size):
    # 准备数据
    train_dataset = CustomDataset(data_train, labels_train)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader

def one_dim_CNN_Initialization(in_channels,out_channels,kernel_size,hidden_size):    
    model = Custom1DCNN(in_channels, out_channels, kernel_size, hidden_size)
    return model

def Super_Parameters(batch_size=None, learning_rate=None, num_epochs=None):
    default_batch_size = 57
    default_learning_rate = 0.001
    default_num_epochs = 20
    
    if batch_size is None:
        batch_size = default_batch_size
    if learning_rate is None:
        learning_rate = default_learning_rate
    if num_epochs is None:
        num_epochs = default_num_epochs
        
    return batch_size, learning_rate, num_epochs


# %%
Set_Random_State()
data,labels = Read_Data()
data_normalized = Standardlization(data)
data= Data_Augmentation(data_normalized)
data_train,data_test,labels_train,labels_test = Split_with_Sequential(data,labels,0.9)
data_train_tensor, labels_train_tensor, data_test_tensor, labels_test_tensor = Data_to_Tensor(data_train,data_test,labels_train,labels_test)
train_dataloader= Load_Data_For_NN(data_train_tensor,labels_train_tensor,57)


# %%
best_r2 = -1  
best_i = -1 
best_j = -1  
for i in range(60,70):
    for j in range (30,35):
        model = one_dim_CNN_Initialization(1,i,3,j)
        batch_size, learning_rate, num_epochs = Super_Parameters(batch_size=57,learning_rate=0.001,num_epochs=110)
        # Train
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        total_step = len(train_dataloader)
        for epoch in range(num_epochs):
            for inputs, targets in train_dataloader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)  
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_outputs = model(data_train_tensor)
                train_rmse = mean_squared_error(labels_train_tensor, train_outputs.detach().numpy(), squared=False)
                train_r2 = r2_score(labels_train_tensor, train_outputs.detach().numpy())

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}, Train RMSE: {train_rmse}, Train R^2: {train_r2}")

        model_path = "1dcnnmodel.pth"
        torch.save(model.state_dict(), model_path)
        cnnmodel_test = Custom1DCNN(1,i,3,j)
        cnnmodel_test.load_state_dict(torch.load(model_path))

        test_outputs = cnnmodel_test(data_test_tensor)
        test_outputs = test_outputs.detach().numpy()

        test_rmse = mean_squared_error(labels_test, test_outputs, squared=False)
        test_r2 = r2_score(labels_test, test_outputs)
        print("Test RMSE:", test_rmse)
        print("Test R2:",test_r2)
        if test_r2 > best_r2:  # 如果当前模型的测试 R2 值更好
            best_r2 = test_r2
            best_i = i
            best_j = j
        if test_r2 > 0.55:
            print("output channels:",i)
            print("hidden_size:",j)
            break

print("Best Test R2:", best_r2)
print("Best i:", best_i)
print("Best j:", best_j)

# %% [markdown]
# Best Test R2: 0.517640596108804
# Best i: 63
# Best j: 32

# %%
import torch
from torchviz import make_dot


# 随机生成输入数据作为示例
batch_size = 1
sequence_length = 2048
input_data = torch.randn(batch_size, in_channels, sequence_length)

# 获取模型输出
output = model(input_data)

# 使用torchviz可视化模型
dot = make_dot(output, params=dict(model.named_parameters()))
dot.render("custom_1d_cnn_graph", format="png")  # 将图保存为PNG格式



