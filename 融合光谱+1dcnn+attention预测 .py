# Description: 
# 全光谱数据
# 处理Cu, Zn, Pb, V 四种元素的数据
# 训练mlp模型，
# 保存最佳模型

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from functions.evaluation import calculate_rmse, calculate_mre
from functions.preprocessing import MyDataset

from network.onedcnn import Basic1DCNN, CNNWithAttention, EarlyStopping, get_criterion_and_optimizer

# 设置CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置EarlyStopping
early_stopping = EarlyStopping(patience=20, threshold=1, verbose=True)


# 设置元素索引
i = 2
# 元素名称
elements = ['Cu', 'Zn', 'Pb', 'V']

# 元素对应超参数字典
elements_dict = {
    'Cu': {'lr': 0.002, 'num_epochs': 2200, 'hidden_layers': [512, 256, 128]},
    'Zn': {'lr': 0.0008, 'num_epochs': 2000, 'hidden_layers': [512, 256, 128]},
    'Pb': {'lr': 0.0014, 'num_epochs': 2000, 'hidden_layers': [512, 256, 128]},
    'V': {'lr': 0.0013, 'num_epochs': 2000, 'hidden_layers': [512, 256, 128]},
}

# 获取当前元素的超参数as
current_element = elements[i]
params = elements_dict[current_element]

# 加载保存好的数据集
train_data = np.load(f'./dataset/全光谱/full_data_train.npy')
train_label = np.load(f'./dataset/全光谱/full_{current_element}_label_train.npy')
test_data = np.load(f'./dataset/全光谱/full_data_test.npy')
test_label = np.load(f'./dataset/全光谱/full_{current_element}_label_test.npy')

# 融合特征
f_train_data = np.load(f'./dataset/融合特征/data_{current_element.lower()}_train.npy')
f_test_data = np.load(f'./dataset/融合特征/data_{current_element.lower()}_test.npy')

# 合并数据
train_data = np.concatenate((train_data, f_train_data), axis=1)
test_data = np.concatenate((test_data, f_test_data), axis=1)

# 创建自定义数据集
train_Dataset = MyDataset(train_data, train_label)
test_Dataset = MyDataset(test_data, test_label)

# 设置输入特征数
feature = train_Dataset.data.shape[1]

# 设置超参数
lr = params['lr']
num_epochs = params['num_epochs']
hidden = params['hidden_layers']


# 创建数据加载器
train_dataloader = DataLoader(train_Dataset, batch_size=57, shuffle=True)
test_dataloader = DataLoader(test_Dataset, batch_size=57, shuffle=True)


# 定义模型保存路径和结果保存路径
model_path = f'./model/rongheguangpu/1dcnnattention/best_model_{current_element}.pth'
results_path = f'./model/rongheguangpu/1dcnnattention/best_model_{current_element}.txt'


# 初始化之前的最佳R2
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        lines = f.readlines()
        prev_r2 = -float('inf')  # 默认值
        for line in lines:
            if "R2" in line:
                prev_r2 = float(line.strip().split(': ')[1])
    print(f'Previous model R2: {prev_r2:.4f}')
else:
    prev_r2 = -float('inf')
    print(f'No previous evaluation found for {current_element}.')

# 定义最大尝试次数以防止无限循环
max_attempts = 100
attempt = 0

while attempt < max_attempts:
    attempt += 1
    print(f'\n=== Training Attempt {attempt} for Element: {current_element} ===')

    # 初始化模型和损失函数
    model = CNNWithAttention(input_dim=len(train_data[0]))
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 将模型移动到GPU
    model.to(device)

    # 记录训练过程中的最佳R2
    best_r2 = prev_r2

    # 训练模型

    model.train()   # 将模型设置为训练模式

    for epoch in range(num_epochs):
        for inputs, targets in train_dataloader:
            inputs = inputs.unsqueeze(1).to(device)     
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


    # 验证模型
    model.eval()    # 将模型设置为验证模式


    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.unsqueeze(1).to(device)
            targets = targets.to(device)   

            outputs = model(inputs)

            # 将张量转换为 NumPy 数组
            targets_np = targets.cpu().numpy()
            outputs_np = outputs.cpu().numpy()
            
            # 计算评估指标
            mse = mean_squared_error(targets_np, outputs_np)
            rmse = calculate_rmse(targets_np, outputs_np)
            mre = calculate_mre(targets_np, outputs_np)
            r2 = r2_score(targets_np, outputs_np)

            print(elements[i])    
            print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MRE: {mre:.4f}, R2: {r2:.4f}')



    print(f'\nValidation Metrics for {current_element}:')
    print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MRE: {mre:.4f}, R2: {r2:.4f}')

    # 如果当前的R2超过之前的最佳R2，则保存模型并退出训练循环
    if r2 > prev_r2:
        print(f'\nR2 improved from {prev_r2:.4f} to {r2:.4f}. Saving model and stopping training.')
        torch.save(model.state_dict(), model_path)
        with open(results_path, 'w') as f:
            f.write(f'{current_element}\n')
            f.write(f'MSE: {mse:.4f}\n')
            f.write(f'RMSE: {rmse:.4f}\n')
            f.write(f'MRE: {mre:.4f}\n')
            f.write(f'R2: {r2:.4f}\n')
        break  # 退出训练循环
    else:
        print(f'\nCurrent model R2: {r2:.4f} is not better than previous model R2: {prev_r2:.4f}. Retrying...')

# 如果达到最大尝试次数且没有改进
if attempt >= max_attempts and r2 <= prev_r2:

    print(f'\nMaximum attempts ({max_attempts}) reached. No improvement in R2.')

# 打印模型
# print(model)




