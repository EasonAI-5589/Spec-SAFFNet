# Description: 使用SVR模型对全光谱数据进行回归预测

import os
import numpy as np
import torch
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from functions.evaluation import calculate_rmse, calculate_mre

# 设置元素索引
i = 3
elements = ['Cu', 'Zn', 'Pb', 'V']
current_element = elements[i]

# SVR参数设置（可根据元素不同微调）
svr_params = {
    'Cu': {'kernel': 'rbf', 'C': 4000, 'gamma': 0.01},
    'Zn': {'kernel': 'rbf', 'C': 6000, 'gamma': 0.008},
    'Pb': {'kernel': 'rbf', 'C': 5000, 'gamma': 0.02},
    'V':  {'kernel': 'rbf', 'C': 100, 'gamma': 0.015}
}
params = svr_params[current_element]

# 加载数据
train_data = np.load(f'./dataset/全光谱/full_data_train.npy')
train_label = np.load(f'./dataset/全光谱/full_{current_element}_label_train.npy').ravel()
test_data = np.load(f'./dataset/全光谱/full_data_test.npy')
test_label = np.load(f'./dataset/全光谱/full_{current_element}_label_test.npy').ravel()

# 初始化SVR模型
model = SVR(kernel=params['kernel'], C=params['C'], gamma=params['gamma'])

# 训练模型
model.fit(train_data, train_label)

# 测试并评估
predictions = model.predict(test_data)

mse = mean_squared_error(test_label, predictions)
rmse = calculate_rmse(test_label, predictions)
mre = calculate_mre(test_label, predictions)
r2 = r2_score(test_label, predictions)

print(current_element)
print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MRE: {mre:.4f}, R2: {r2:.4f}')
# 设置模型和结果保存路径
model_dir = f'./model/quanguangpu/svr'
os.makedirs(model_dir, exist_ok=True)  # 如果目录不存在则创建

model_path = os.path.join(model_dir, f'best_model_{current_element}.pth')
results_path = os.path.join(model_dir, f'best_model_{current_element}.txt')

# 如果模型文件不存在，直接保存当前模型和评估结果
if not os.path.exists(model_path):
    torch.save(model, model_path)
    with open(results_path, 'w') as f:
        f.write(f'{elements[i]}\n')
        f.write(f'MSE: {mse:.4f}\n')
        f.write(f'RMSE: {rmse:.4f}\n')
        f.write(f'MRE: {mre:.4f}\n')
        f.write(f'R2: {r2:.4f}\n')
    print(f'No previous model found. Current model saved as the best with R2: {r2:.4f}')

else:
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

    # 比较当前模型和旧模型的R²
    if r2 > prev_r2:
        torch.save(model, model_path)
        with open(results_path, 'w') as f:
            f.write(f'{elements[i]}\n')
            f.write(f'MSE: {mse:.4f}\n')
            f.write(f'RMSE: {rmse:.4f}\n')
            f.write(f'MRE: {mre:.4f}\n')
            f.write(f'R2: {r2:.4f}\n')
        print(f'New best model saved with R2: {r2:.4f}')
    else:
        print(f'Current model R2: {r2:.4f} is not better than previous model R2: {prev_r2:.4f}')