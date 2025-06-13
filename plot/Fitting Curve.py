#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import ast
import numpy as np
import matplotlib.pyplot as plt

# ——— 配置 ———
element = 'Cu'  # 可改为 'Cu','Zn','Pb','V'

# ——— 路径解析 ———
# 当前脚本所在目录（plot/）
script_dir = os.path.dirname(os.path.abspath(__file__))
# 项目根目录（plot/ 上一级）
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
# 存放评估结果的目录
results_dir = os.path.join(project_root, 'best_results')
# 要读取的 txt 文件
file_path = os.path.join(results_dir, f'evaluation_{element}.txt')

# ——— 读取并解析 txt ———
with open(file_path, 'r') as f:
    content = f.read()

# 提取评估指标
mse  = float(re.search(r'MSE:\s*([\d\.]+)', content).group(1))
rmse = float(re.search(r'RMSE:\s*([\d\.]+)', content).group(1))
mre  = float(re.search(r'MRE:\s*([\d\.]+)', content).group(1))
r2   = float(re.search(r'R2:\s*([\d\.]+)', content).group(1))

# 提取 Target 数组文本块
target_block = re.search(
    r'==== Target concentrations for \w+ ====\n(.+?)\n\n',
    content,
    re.DOTALL
).group(1)

# 提取 Predicted 数组文本块
pred_block = re.search(
    r'==== Predicted concentrations for \w+ ====\n(.+)',
    content,
    re.DOTALL
).group(1)

# 转为 NumPy 数组
all_targets = np.array(ast.literal_eval(target_block))
all_preds   = np.array(ast.literal_eval(pred_block))

# 将 (n,1) → (n,)
targets = all_targets.flatten()
preds   = all_preds.flatten()

# ——— 绘制 拟合程度 折线图 ———
n = len(targets)
x = np.arange(n)

plt.figure(figsize=(8, 6))

plt.plot(x, targets, color='red',   marker='o', label='True Value')
plt.plot(x, preds,   color='blue',  marker='o', label='Predicted Value')

# 主刻度：每个样本，次刻度：自动
plt.xticks(x)
plt.minorticks_on()

# y 轴刻度分成 6 段
ymin = min(targets.min(), preds.min())
ymax = max(targets.max(), preds.max())
plt.yticks(np.linspace(ymin, ymax, 6))

# 网格：主格与次格
plt.grid(which='major', linestyle='-',  linewidth=0.8, alpha=0.8)
plt.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.5)

plt.xlabel('Sample Number')
plt.ylabel('Concentration')
plt.title(f'Comparison of True and Predicted Values for {element}\n'
          f'MSE={mse:.2f}, RMSE={rmse:.2f}, R²={r2:.4f}')
plt.legend()
plt.tight_layout()

# ——— 保存为 PDF ———
out_dir = os.path.join(project_root, 'output_plot')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'fitting_curve_{element}.pdf')
plt.savefig(out_path, format='pdf', bbox_inches='tight')
print(f'Fitting curve saved to: {out_path}')

plt.show()