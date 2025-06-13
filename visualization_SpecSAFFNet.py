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
i = 3
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


# -------------------------------
# 加载最佳模型并评估
# -------------------------------
model_type = 'attention'
best_model_path = f'./model/rongheguangpu/1dcnnattention/best_model_{current_element}.pth'

# 实例化模型结构
if model_type == 'attention':
    best_model = CNNWithAttention(input_dim=len(train_data[0]))
else:
    best_model = Basic1DCNN()

# 加载权重
state_dict = torch.load(best_model_path, map_location=device)
best_model.load_state_dict(state_dict)

# 设置评估模式
best_model.to(device)
best_model.eval()

# 执行评估
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_dataloader:
        inputs = inputs.unsqueeze(1).to(device)
        targets = targets.to(device)
        outputs = best_model(inputs)

        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

mse = mean_squared_error(all_targets, all_preds)
rmse = calculate_rmse(all_targets, all_preds)
mre = calculate_mre(all_targets, all_preds)
r2 = r2_score(all_targets, all_preds)

# 打印结果
print(f'== Evaluating best model for {current_element} ==')
print(f'MSE:  {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MRE:  {mre:.4f}')
print(f'R2:   {r2:.4f}')
print(f"==== Target concentrations ====\n{all_targets}")
print(f"==== Predicted concentrations ====\n{all_preds}")

# 保存结果
output_dir = './best_results/rongheguangpu'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f'evaluation_{current_element}.txt')

with open(output_path, 'w') as f:
    f.write(f'== Evaluating best model for {current_element} ==\n')
    f.write(f'MSE:  {mse:.4f}\n')
    f.write(f'RMSE: {rmse:.4f}\n')
    f.write(f'MRE:  {mre:.4f}\n')
    f.write(f'R2:   {r2:.4f}\n\n')
    f.write(f'==== Target concentrations for {current_element} ====\n')
    f.write(np.array2string(all_targets, precision=4, separator=', ') + '\n\n')
    f.write(f'==== Predicted concentrations for {current_element} ====\n')
    f.write(np.array2string(all_preds, precision=4, separator=', ') + '\n')

print(f'评估结果已保存到: {output_path}')