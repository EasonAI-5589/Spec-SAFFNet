import os
import torch

import os
import torch

def evaluate_and_save_model(model, current_element, r2, mse, rmse, mre, model_path, results_path):
    """
    评估模型并根据 R² 分数决定是否保存新模型和结果。

    参数:
        model: 需要评估的模型。
        current_element: 当前元素名称，用于文件命名和输出。
        r2: 当前模型的 R² 分数。
        mse: 当前模型的均方误差。
        rmse: 当前模型的均方根误差。
        mre: 当前模型的相对误差。
        model_path: 模型保存路径。
        results_path: 结果保存路径。
    """
    
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
            f.write(f'{current_element}\n')
            f.write(f'MSE: {mse:.4f}\n')
            f.write(f'RMSE: {rmse:.4f}\n')
            f.write(f'MRE: {mre:.4f}\n')
            f.write(f'R2: {r2:.4f}\n')
    else:
        print(f'Current model R2: {r2:.4f} is not better than previous model R2: {prev_r2:.4f}')

# 示例调用
# evaluate_and_save_model(model, current_element, r2, mse, rmse, mre, model_path, results_path)
