import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error, r2_score

from functions.evaluation import calculate_rmse, calculate_mre
from functions.preprocessing import MyDataset
from functions.save_info import evaluate_and_save_model

from network.mlp import MLP, MLP_3_hidden

# 设置CUDA设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置元素索引
i = 2
# 元素名称
elements = ['Cu', 'Zn', 'Pb', 'V']

# 元素对应超参数字典
elements_dict = {
    'Cu': {'model': 'MLP_3_hidden','lr': 0.002, 'num_epochs': 2000, 'hidden_layers': [512, 256, 128]},
    'Zn': {'model': 'MLP_3_hidden','lr': 0.0015, 'num_epochs': 1800, 'hidden_layers': [512, 256, 128]},
    'Pb': {'model': 'MLP_3_hidden','lr': 0.0015, 'num_epochs': 1800, 'hidden_layers': [512, 256, 128]},
    'V': {'model': 'MLP_3_hidden','lr': 0.0005, 'num_epochs': 2100, 'hidden_layers': [32, 16, 4]},
}


# 获取当前元素的超参数as
current_element = elements[i]
current_element_lower = current_element.lower()
params = elements_dict[current_element]

# 加载保存好的数据集
train_data = np.load(f'./dataset/特征选择/data_{current_element_lower}_train.npy')
train_label = np.load(f'./dataset/全光谱/full_{current_element}_label_train.npy')
test_data = np.load(f'./dataset/特征选择/data_{current_element_lower}_test.npy')
test_label = np.load(f'./dataset/全光谱/full_{current_element}_label_test.npy')

# 创建自定义数据集
train_Dataset = MyDataset(train_data, train_label)
test_Dataset = MyDataset(test_data, test_label)

# 设置输入特征数
feature = train_Dataset.data.shape[1]

# 设置超参数
model = params['model']
lr = params['lr']
num_epochs = params['num_epochs']
hidden = params['hidden_layers']


# 创建数据加载器
train_dataloader = DataLoader(train_Dataset, batch_size=57, shuffle=True)
test_dataloader = DataLoader(test_Dataset, batch_size=57, shuffle=True)


# 初始化模型和损失函数

if model == 'MLP':
    model = MLP(feature, hidden[0], hidden[1])
elif model == 'MLP_3_hidden':
    model = MLP_3_hidden(feature, hidden[0], hidden[1],hidden[2])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

model.to(device)

# 训练模型

model.train()   # 将模型设置为训练模式

for epoch in range(num_epochs):
    for inputs, targets in train_dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

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
        inputs, targets = inputs.to(device), targets.to(device)

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



# 读取之前的模型评估结果
model_path = f'./model/quanguangpu/feature_selection/best_model_{current_element}.pth'
results_path = f'./model/quanguangpu/feature_selection/best_model_{current_element}.txt'

evaluate_and_save_model(model, current_element, r2, mse, rmse, mre, model_path, results_path)

print(model)










