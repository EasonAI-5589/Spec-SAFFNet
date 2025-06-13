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
    'Cu': {'lr': 0.005, 'num_epochs': 3000, 'hidden_layers': [512, 256, 128]},
    'Zn': {'lr': 0.004, 'num_epochs': 800, 'hidden_layers': [512, 256, 128]},
    'Pb': {'lr': 0.005, 'num_epochs': 2500, 'hidden_layers': [512, 256, 128]},
    'V': {'lr': 0.0015, 'num_epochs': 2500, 'hidden_layers': [512, 256, 128]},
}

# 获取当前元素的超参数as
current_element = elements[i]
params = elements_dict[current_element]

# 加载保存好的数据集
train_data = np.load(f'./dataset/全光谱/full_data_train.npy')
train_label = np.load(f'./dataset/全光谱/full_{current_element}_label_train.npy')
test_data = np.load(f'./dataset/全光谱/full_data_test.npy')
test_label = np.load(f'./dataset/全光谱/full_{current_element}_label_test.npy')

# 创建自定义数据集
train_Dataset = MyDataset(train_data, train_label)
test_Dataset = MyDataset(test_data, test_label)

# 创建数据加载器
train_dataloader = DataLoader(train_Dataset, batch_size=57, shuffle=True)
test_dataloader = DataLoader(test_Dataset, batch_size=57, shuffle=True)


# 初始化模型和损失函数
# 模型选择
model_type = 'attention'  # 或 'basic'
if model_type == 'basic':
    model = Basic1DCNN()
else:
    model = CNNWithAttention()


# -------------------------------
# 仅加 载 最佳 模型 并 评估
# -------------------------------

best_model_path = f'./model/rongheguangpu/1dcnnattention/best_model_{current_element}.pth'
best_model = torch.load(best_model_path, map_location=device)

# 1) 重新实例化模型结构
if model_type == 'basic':
    best_model = Basic1DCNN()
else:
    best_model = CNNWithAttention()

# 2) 加载权重
state_dict = torch.load(best_model_path, map_location=device)
best_model.load_state_dict(state_dict)

# 3) 移到 device，切换到 eval 模式
best_model.to(device)
best_model.eval()

# 4) 在测试集上跑一遍
all_preds = []
all_targets = []

with torch.no_grad():
    for inputs, targets in test_dataloader:
        # 调整维度并移动到 GPU/CPU
        inputs = inputs.unsqueeze(1).to(device)  # (batch, 1, features)
        targets = targets.to(device)

        outputs = best_model(inputs)

        all_preds.append(outputs.cpu().numpy())
        all_targets.append(targets.cpu().numpy())

# 5) 拼接并计算指标
all_preds = np.concatenate(all_preds, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

mse  = mean_squared_error(all_targets, all_preds)
rmse = calculate_rmse(all_targets, all_preds)
mre  = calculate_mre(all_targets, all_preds)
r2   = r2_score(all_targets, all_preds)

# 6) 打印结果
print(f'== Evaluating best model for {current_element} ==')
print(f'MSE:  {mse:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MRE:  {mre:.4f}')
print(f'R2:   {r2:.4f}')

print(f"==== Target concentrations for {current_element} ====")
print(all_targets)
print(f"==== Predicted concentrations for {current_element} ====")
print(all_preds)

# 确保输出目录存在
output_dir = './best_results/rongheguangpu'
os.makedirs(output_dir, exist_ok=True)

# 定义输出文件路径
output_path = os.path.join(output_dir, f'evaluation_{current_element}.txt')

# 写入文件
with open(output_path, 'w') as f:
    # 写评估指标
    f.write(f'== Evaluating best model for {current_element} ==\n')
    f.write(f'MSE:  {mse:.4f}\n')
    f.write(f'RMSE: {rmse:.4f}\n')
    f.write(f'MRE:  {mre:.4f}\n')
    f.write(f'R2:   {r2:.4f}\n\n')

    # 写目标值
    f.write(f'==== Target concentrations for {current_element} ====\n')
    # 将数组格式化为字符串（精度 4 位，小数点后用逗号分隔）
    f.write(np.array2string(all_targets, precision=4, separator=', ') + '\n\n')

    # 写预测值
    f.write(f'==== Predicted concentrations for {current_element} ====\n')
    f.write(np.array2string(all_preds, precision=4, separator=', ') + '\n')

print(f'评估结果已保存到: {output_path}')





