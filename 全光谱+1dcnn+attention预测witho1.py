# Description:
# 全光谱数据处理，针对 Cu, Zn, Pb, V 四种元素
# 训练 MLP 模型，保存最佳模型

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import logging
import json
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score

from functions.evaluation import calculate_rmse, calculate_mre
from functions.preprocessing import MyDataset
from network.onedcnn import Basic1DCNN, CNNWithAttention, EarlyStopping, get_criterion_and_optimizer

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_data(element, data_dir='./dataset/全光谱/'):
    train_data = np.load(os.path.join(data_dir, 'full_data_train.npy'))
    train_label = np.load(os.path.join(data_dir, f'full_{element}_label_train.npy'))
    test_data = np.load(os.path.join(data_dir, 'full_data_test.npy'))
    test_label = np.load(os.path.join(data_dir, f'full_{element}_label_test.npy'))
    return train_data, train_label, test_data, test_label


def get_hyperparameters(element, elements_dict):
    return elements_dict[element]


def create_dataloaders(train_data, train_label, test_data, test_label, batch_size=57):
    train_dataset = MyDataset(train_data, train_label)
    test_dataset = MyDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def initialize_model(model_type, hidden_layers, input_size):
    if model_type == 'basic':
        model = Basic1DCNN(input_size, hidden_layers)
    elif model_type == 'attention':
        model = CNNWithAttention(input_size, hidden_layers)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return model


def train_model(model, criterion, optimizer, train_loader, early_stopping, device, num_epochs):
    model.train()
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        for inputs, targets in train_loader:
            inputs = inputs.unsqueeze(1).to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}')

        # 早停检查
        early_stopping(avg_loss)
        if early_stopping.early_stop:
            logger.info(f'Early stopping triggered at epoch {epoch}')
            break


def evaluate_model(model, test_loader, device):
    model.eval()
    all_targets = []
    all_outputs = []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.unsqueeze(1).to(device)
            targets = targets.to(device)
            outputs = model(inputs)

            all_targets.extend(targets.cpu().numpy())
            all_outputs.extend(outputs.cpu().numpy())

    mse = mean_squared_error(all_targets, all_outputs)
    rmse = calculate_rmse(np.array(all_targets), np.array(all_outputs))
    mre = calculate_mre(np.array(all_targets), np.array(all_outputs))
    r2 = r2_score(all_targets, all_outputs)

    return mse, rmse, mre, r2


def save_results(element, mse, rmse, mre, r2, results_path):
    result = {
        'element': element,
        'MSE': mse,
        'RMSE': rmse,
        'MRE': mre,
        'R2': r2
    }
    with open(results_path, 'w') as f:
        json.dump(result, f, indent=4)
    logger.info(f'Results saved to {results_path}')


def load_previous_r2(results_path):
    if os.path.exists(results_path):
        with open(results_path, 'r') as f:
            result = json.load(f)
            return result.get('R2', -float('inf'))
    return -float('inf')


def save_model_if_better(model, r2, prev_r2, model_path, results_path, mse, rmse, mre):
    if r2 > prev_r2:
        torch.save(model.state_dict(), model_path)
        logger.info(f'New best model saved with R2: {r2:.4f}')
        save_results(model_path.split('/')[-1].replace('.pth', ''), mse, rmse, mre, r2, results_path)
    else:
        logger.info(f'Current model R2: {r2:.4f} is not better than previous model R2: {prev_r2:.4f}')


def main():
    # 配置
    elements = ['Cu', 'Zn', 'Pb', 'V']
    elements_dict = {
        'Cu': {'lr': 0.005, 'num_epochs': 3000, 'hidden_layers': [512, 256, 128]},
        'Zn': {'lr': 0.004, 'num_epochs': 3000, 'hidden_layers': [512, 256, 128]},
        'Pb': {'lr': 0.005, 'num_epochs': 2500, 'hidden_layers': [512, 256, 128]},
        'V': {'lr': 0.0015, 'num_epochs': 2500, 'hidden_layers': [512, 256, 128]},
    }
    model_type = 'attention'  # 或 'basic'
    data_dir = './dataset/全光谱/'
    model_save_dir = './model/quanguangpu/onedcnnattention/'
    os.makedirs(model_save_dir, exist_ok=True)
    batch_size = 57

    # 设置CUDA设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')

    for i, element in enumerate(elements):
        logger.info(f'Processing element: {element}')

        # 获取超参数
        params = get_hyperparameters(element, elements_dict)
        lr = params['lr']
        num_epochs = params['num_epochs']
        hidden_layers = params['hidden_layers']

        # 加载数据
        train_data, train_label, test_data, test_label = load_data(element, data_dir)

        # 创建数据加载器
        train_loader, test_loader = create_dataloaders(train_data, train_label, test_data, test_label, batch_size)

        # 获取输入特征数
        input_size = train_data.shape[1]

        # 初始化模型
        model = initialize_model(model_type, hidden_layers, input_size).to(device)
        logger.info(f'Model: {model}')

        # 获取损失函数和优化器
        criterion, optimizer = get_criterion_and_optimizer(model, lr=lr)

        # 设置早停
        early_stopping = EarlyStopping(patience=20, threshold=1, verbose=True)

        # 训练模型
        train_model(model, criterion, optimizer, train_loader, early_stopping, device, num_epochs)

        # 评估模型
        mse, rmse, mre, r2 = evaluate_model(model, test_loader, device)
        logger.info(f'Element: {element}, MSE: {mse:.4f}, RMSE: {rmse:.4f}, MRE: {mre:.4f}, R2: {r2:.4f}')

        # 处理模型保存
        model_path = os.path.join(model_save_dir, f'best_model_{element}.pth')
        results_path = os.path.join(model_save_dir, f'best_model_{element}.json')
        prev_r2 = load_previous_r2(results_path)
        logger.info(f'Previous model R2: {prev_r2:.4f}')
        save_model_if_better(model, r2, prev_r2, model_path, results_path, mse, rmse, mre)


if __name__ == '__main__':
    main()
