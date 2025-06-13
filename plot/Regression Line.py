#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import ast
import numpy as np
import matplotlib.pyplot as plt

# ——— 配置 ———
element = 'Pb'  # 可改为 'Cu','Zn','Pb','V'

# ——— 路径解析 ———
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
results_dir  = os.path.join(project_root, 'best_results')
file_path    = os.path.join(results_dir, f'evaluation_{element}.txt')

# ——— 读取并解析 txt ———
with open(file_path, 'r') as f:
    content = f.read()

mse  = float(re.search(r'MSE:\s*([\d\.]+)', content).group(1))
rmse = float(re.search(r'RMSE:\s*([\d\.]+)', content).group(1))
mre  = float(re.search(r'MRE:\s*([\d\.]+)', content).group(1))
r2   = float(re.search(r'R2:\s*([\d\.]+)', content).group(1))

target_block = re.search(r'==== Target concentrations.*?\n(.+?)\n\n',
                         content, re.DOTALL).group(1)
pred_block   = re.search(r'==== Predicted concentrations.*?\n(.+)',
                         content, re.DOTALL).group(1)

# 转 NumPy 并 flatten
all_targets = np.array(ast.literal_eval(target_block)).flatten()
all_preds   = np.array(ast.literal_eval(pred_block)).flatten()

# ——— 绘制 R² 散点图 ———
fig, ax = plt.subplots(figsize=(6, 6))           # 正方形画布

# 1) 散点
ax.scatter(
    all_targets, all_preds,
    s=60,
    edgecolors='tab:blue',
    facecolors='none',
    marker='o',
    linewidths=1.2,
    label='Predicted vs True'
)

# 2) 理想对角线
minv = min(all_targets.min(), all_preds.min())
maxv = max(all_targets.max(), all_preds.max())
# 扩一点边距
pad = 0.05 * (maxv - minv)
ax.plot(
    [minv-pad, maxv+pad], [minv-pad, maxv+pad],
    color='red', linewidth=2,
    label='Ideal Fit'
)

# 3) 刻度与网格
ax.set_xlim(minv-pad, maxv+pad)
ax.set_ylim(minv-pad, maxv+pad)
ax.set_xticks(np.linspace(minv, maxv, 6))     # 主刻度 6 分段
ax.set_yticks(np.linspace(minv, maxv, 6))
ax.minorticks_on()                            # 打开次刻度

ax.grid(which='major', linestyle='-',  linewidth=0.8, alpha=0.7)
ax.grid(which='minor', linestyle='--', linewidth=0.5, alpha=0.4)

# 4) 强制坐标区为正方形
ax.set_aspect('equal', 'box')

# 5) 标签与图例
ax.set_xlabel('True Values')
ax.set_ylabel('Predicted Values')
ax.set_title(f'Predictions vs True (R\u00b2 = {r2:.4f})')
ax.legend(frameon=False, loc='upper left')

plt.tight_layout()

# 6) 保存为 PDF
out_dir = os.path.join(project_root, 'output_plot')
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f'regression_scatter_{element}.pdf')
fig.savefig(out_path, format='pdf', bbox_inches='tight')
print(f'Saved scatter to {out_path}')

plt.show()