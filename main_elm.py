# Description: 
# 专家数据，
# 处理Cu, Zn, Pb, V, Co五种元素的数据
# 训练ELM模型，
# 保存最佳模型

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

from functions.preprocessing import MyDataset
from network.elm import ELM


# 设置元素索引
i = 3

# 元素名称
elements = ['Cu', 'Zn', 'Pb', 'V', 'Co']

# 元素对应超参数字典
elements_dict = {
    'Cu': {'lr': 0.001, 'num_epochs': 4500, 'hidden_layers': 32},
    'Zn': {'lr': 0.0012, 'num_epochs': 4500, 'hidden_layers': 32},
    'Pb': {'lr': 0.0015, 'num_epochs': 8000, 'hidden_layers': 32},
    'V': {'lr': 0.0013, 'num_epochs': 8000, 'hidden_layers': 20},
    'Co': {'lr': 0.001, 'num_epochs': 2000, 'hidden_layers': 20}
}

# 获取当前元素的超参数
current_element = elements[i]
params = elements_dict[current_element]

# 加载保存好的数据集
with open(f'./dataset/{current_element}_train_Dataset.pkl', 'rb') as f:
    train_Dataset = pickle.load(f)
with open(f'./dataset/{current_element}_test_Dataset.pkl', 'rb') as f:
    test_Dataset = pickle.load(f)

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
model = ELM(feature, hidden)
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


# 定义计算 RMSE 和 MRE 的函数
def calculate_rmse(targets, outputs):
    mse = mean_squared_error(targets, outputs)
    rmse = np.sqrt(mse)
    return rmse

def calculate_mre(targets, outputs):
    relative_errors = np.abs((targets - outputs) / targets)
    mre = np.mean(relative_errors)
    return mre


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


# 比较模型，如果当前模型的 R² 分数更高，则替换之前的模型
model_path = f'./model/elm/best_model_{current_element}.pth'
model_path_r2 = f'./model/elm/best_model_{current_element}_{r2:.4f}.pth'
if os.path.exists(model_path):
    # 加载之前的模型
    previous_model = ELM(feature, hidden)
    previous_model.load_state_dict(torch.load(model_path,weights_only=True))
    previous_model.eval()

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            prev_outputs = previous_model(inputs)
            prev_targets_np = targets.cpu().numpy()
            prev_outputs_np = prev_outputs.cpu().numpy()
            prev_r2 = r2_score(prev_targets_np, prev_outputs_np)

    print(f'Previous model R2: {prev_r2:.4f}')
else:
    prev_r2 = -float('inf')

if r2 > prev_r2:
    torch.save(model.state_dict(), model_path)
    # torch.save(model.state_dict(), model_path_r2)
    print(f'New best model saved with R2: {r2:.4f}')
else:
    print(f'Current model R2: {r2:.4f} is not better than previous model R2: {prev_r2:.4f}')