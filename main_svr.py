# Description



import os
import pickle
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


from functions.preprocessing import MyDataset
from functions.evaluation import calculate_rmse, calculate_mre




# 设置元素索引
i = 0

# 元素名称
elements = ['Cu', 'Zn', 'Pb', 'V', 'Co']

# 元素对应超参数字典
elements_dict = {
    'Cu': {'lr': 0.001, 'num_epochs': 2000, 'hidden_layers': [32, 16]},
    'Zn': {'lr': 0.001, 'num_epochs': 2000, 'hidden_layers': [32, 16]},
    'Pb': {'lr': 0.00125, 'num_epochs': 6000, 'hidden_layers': [32, 16]},
    'V': {'lr': 0.0013, 'num_epochs': 6500, 'hidden_layers': [20, 10]},
    'Co': {'lr': 0.001, 'num_epochs': 2000, 'hidden_layers': [20, 8]}
}

# 获取当前元素的超参数
current_element = elements[i]
params = elements_dict[current_element]

# 加载保存好的数据集
with open(f'./dataset/{current_element}_train_Dataset.pkl', 'rb') as f:
    train_Dataset = pickle.load(f)
with open(f'./dataset/{current_element}_test_Dataset.pkl', 'rb') as f:
    test_Dataset = pickle.load(f)


train_data = train_Dataset.data
train_label = train_Dataset.label.reshape(-1)
test_data = test_Dataset.data
test_label = test_Dataset.label.reshape(-1)

# 创建svr模型

svr = SVR(kernel='rbf', C=1e3, gamma=0.1)

# 训练模型

svr.fit(train_data, train_label)

# 预测

train_pred = svr.predict(train_data)
test_pred = svr.predict(test_data)

# 评估

train_mse = mean_squared_error(train_label, train_pred)
train_r2 = r2_score(train_label, train_pred)
train_rmse = calculate_rmse(train_label, train_pred)
train_mre = calculate_mre(train_label, train_pred)

test_mse = mean_squared_error(test_label, test_pred)
test_r2 = r2_score(test_label, test_pred)
test_rmse = calculate_rmse(test_label, test_pred)
test_mre = calculate_mre(test_label, test_pred)

print(f'{current_element} train mse: {train_mse:.4f}, train r2: {train_r2:.4f}, train_rmse: {train_rmse:.4f}, train_mre: {train_mre:.4f}')
print(f'{current_element} test mse: {test_mse:.4f}, test r2: {test_r2:.4f}, test_rmse: {test_rmse:.4f}, test_mre: {test_mre:.4f}')

