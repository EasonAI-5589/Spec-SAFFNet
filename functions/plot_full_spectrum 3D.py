import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# 读取数据
df = pd.read_excel('./data/full_spectrum.xlsx')

# 获取样本数量和通道数量
samples = df.shape[0]  # 样本数
channels = df.shape[1]  # 每个样本的通道数

# 设置显示的通道数
disp_channels = 600


# 创建3D图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制每个样本的3D光谱图
for sample_id in range(samples):
    # 获取当前样本的通道数据，此处显示前600个通道
    z = df.iloc[sample_id, :disp_channels].values
    
    # x 对应于通道的索引或具体通道值
    x = np.arange(disp_channels)
    
    # y 对应于样本的ID
    y = np.full_like(x, sample_id)
    
    # 绘制每个样本的曲线
    ax.plot(x, y, z, lw=0.5)

# 设置坐标轴标签
ax.set_xlabel('Spectral Channel', fontsize=12)
ax.set_ylabel('Sample ID', fontsize=12)
ax.set_zlabel('Counts (x$ 10^5$)', fontsize=12)

# 设置坐标轴刻度字体大小
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.tick_params(axis='z', labelsize=10)


# 添加图表标题
ax.set_title('Visualization of Spectral Data for Samples', fontsize=14)



# 保存图形
plt.savefig('./plot/spectrum_3D/3D_full_spectrum.png', dpi=400)