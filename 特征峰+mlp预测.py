# Description: 
# 特征峰数据
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

from network.mlp import MLP


# 设置元素索引
i = 3

# 元素名称
elements = ['Cu', 'Zn', 'Pb', 'V']

# 元素对应超参数字典
elements_dict = {
    'Cu': {'lr': 0.001, 'num_epochs': 2000, 'hidden_layers': [32, 16]},
    'Zn': {'lr': 0.001, 'num_epochs': 2000, 'hidden_layers': [32, 16]},
    'Pb': {'lr': 0.002, 'num_epochs': 3000, 'hidden_layers': [32, 16]},
    'V': {'lr': 0.0011, 'num_epochs': 3000, 'hidden_layers': [64, 32]},
}

# 获取当前元素的超参数
current_element = elements[i]
params = elements_dict[current_element]

# 加载保存好的数据集
train_data = np.load(f'./dataset/特征峰/{current_element}_train_data.npy')
train_label = np.load(f'./dataset/特征峰/{current_element}_train_label.npy')
test_data = np.load(f'./dataset/特征峰/{current_element}_test_data.npy')
test_label = np.load(f'./dataset/特征峰/{current_element}_test_label.npy')

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


# 初始化模型和损失函数
model = MLP(feature, hidden[0], hidden[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# 训练模型

model.train()   # 将模型设置为训练模式

for epoch in range(num_epochs):
    for inputs, targets in train_dataloader:
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
        outputs = model(inputs)

        # 将张量转换为 NumPy 数组
        targets_np = targets.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        
        mse = mean_squared_error(targets_np, outputs_np)
        rmse = calculate_rmse(targets_np, outputs_np)
        mre = calculate_mre(targets_np, outputs_np)
        r2 = r2_score(targets_np, outputs_np)

        print(elements[i])    
        print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MRE: {mre:.4f}, R2: {r2:.4f}')


# 加载之前的模型评估结果

# 读取之前的模型评估结果
model_path = f'./model/tezhengfeng/mlp/best_model_{current_element}.pth'
results_path = f'./model/tezhengfeng/mlp/best_model_{current_element}.txt'

# 初始化最大R²值
prev_r2 = -float('inf')

# 如果存在结果文件，则读取之前的R²
if os.path.exists(results_path):
    with open(results_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "R2" in line:
                prev_r2 = float(line.strip().split(': ')[1])
    print(f'Previous model R2: {prev_r2:.4f}')
else:
    print(f'No previous evaluation found for {current_element}.')

# 比较模型，如果当前模型的 R² 分数更高，则替换之前的模型和评估结果
if r2 > prev_r2:
    # 保存新的模型权重
    torch.save(model.state_dict(), model_path)
    print(f'New best model saved with R2: {r2:.4f}')
    
    # 保存新的评估结果
    with open(results_path, 'w') as f:
        f.write(f'{elements[i]}\n')
        f.write(f'MSE: {mse:.4f}\n')
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'MRE: {mre:.4f}\n')
        f.write(f'R2: {r2:.4f}\n')
else:
    print(f'Current model R2: {r2:.4f} is not better than previous model R2: {prev_r2:.4f}')







