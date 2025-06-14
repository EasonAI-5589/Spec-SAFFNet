# Description: 使用PLS回归模型对全光谱数据进行回归预测

import os
import numpy as np
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from functions.evaluation import calculate_rmse, calculate_mre

# 设置元素索引
i = 2  # 可修改为 0,1,2,3 分别对应 Cu, Zn, Pb, V
elements = ['Cu', 'Zn', 'Pb', 'V']
current_element = elements[i]

# 设置PLS回归参数（可按元素调整）
pls_params = {
    'Cu': {'n_components': 15},
    'Zn': {'n_components': 20},
    'Pb': {'n_components': 25},
    'V':  {'n_components': 12}
}
params = pls_params[current_element]

# 加载数据
train_data = np.load(f'./dataset/全光谱/full_data_train.npy')
train_label = np.load(f'./dataset/全光谱/full_{current_element}_label_train.npy').ravel()
test_data = np.load(f'./dataset/全光谱/full_data_test.npy')
test_label = np.load(f'./dataset/全光谱/full_{current_element}_label_test.npy').ravel()

# 初始化PLS模型
model = PLSRegression(n_components=params['n_components'])

# 训练模型
model.fit(train_data, train_label)

# 测试并评估
predictions = model.predict(test_data).ravel()

mse = mean_squared_error(test_label, predictions)
rmse = calculate_rmse(test_label, predictions)
mre = calculate_mre(test_label, predictions)
r2 = r2_score(test_label, predictions)

print(current_element)
print(f'MSE: {mse:.4f}, RMSE: {rmse:.4f}, MRE: {mre:.4f}, R2: {r2:.4f}')

# 设置模型保存路径
model_dir = f'./model/quanguangpu/pls'
os.makedirs(model_dir, exist_ok=True)

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
    prev_r2 = -float('inf')
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if "R2" in line:
                    prev_r2 = float(line.strip().split(': ')[1])
        print(f'Previous model R2: {prev_r2:.4f}')
    else:
        print(f'No previous evaluation found for {current_element}.')

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