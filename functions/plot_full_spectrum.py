# Function: 绘制全谱图


import os
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('./data/full_spectrum.xlsx')


def draw_full_spectrum_with_inset(sample_id):
    # 读取所有channels的值
    channels = df.iloc[sample_id,:].values

    # print(channels[200])


    # 创建主图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制主图
    ax.plot(channels, label='Full Spectrum Data', color='b')
    ax.set_xlabel('Spectral channels')
    ax.set_ylabel('Counts')
    ax.set_title('Comparison of the original and subtracted background spectra')
    ax.legend()
    ax.grid(True)

    # 创建子图（放大部分区域）
    inset_ax = fig.add_axes([0.4, 0.4, 0.4, 0.4])  # [left, bottom, width, height]
    inset_ax.plot(channels, label='Full Spectrum Data', color='b')
    inset_ax.set_xlim(150, 400)  # 放大区域的x轴范围
    inset_ax.set_ylim(0, 10000)  # 放大区域的y轴范围
    inset_ax.grid(True)

    # 添加注释和箭头
    # ax.annotate('', xy=(200, 0), xytext=(1000, 300000),
    #              arrowprops=dict(facecolor='black', shrink=0.05, width=0.5, headwidth=5))

    # 保存图形为文件
    plt.savefig('./output_plot/spectrum/full_spectrum_with_inset_'+str(sample_id)+'.png', dpi=300)  # 保存为PNG格式，分辨率为300 DPI


def draw_full_spectrum(sample_id):
    # 读取所有channels的值
    channels = df.iloc[sample_id,:].values

    # print(channels[200])


    # 创建主图
    fig, ax = plt.subplots(figsize=(12, 8))

    # 绘制主图
    ax.plot(channels, label='Full Spectrum Data', color='b')
    ax.set_xlabel('Spectral channels')
    ax.set_ylabel('Counts')
    ax.set_title('Comparison of the original and subtracted background spectra')
    ax.legend()
    ax.grid(True)

    # 保存图形为文件
    plt.savefig('./plot/spectrum/full_spectrum_'+str(sample_id)+'.png', dpi=300)  # 保存为PNG格式，分辨率为300 DPI



# 从id=1开始绘制，保存绘制结果1,10,20,30,40,50

flag = True

if flag:
    draw_full_spectrum(1)
    draw_full_spectrum(10)
    draw_full_spectrum(20)
    draw_full_spectrum(30)
    draw_full_spectrum(40)
    draw_full_spectrum(50)
else:
    draw_full_spectrum_with_inset(1)
    draw_full_spectrum_with_inset(10)
    draw_full_spectrum_with_inset(20)
    draw_full_spectrum_with_inset(30)
    draw_full_spectrum_with_inset(40)
    draw_full_spectrum_with_inset(50)

