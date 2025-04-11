import matplotlib.pyplot as plt
import os


def draw_scatter(y1, y2, rmse, r2, save_directory, xlabel, ylabel, title):
    """
    绘制预测值y1和真实值y2的散点图
    """

    plt.clf() # 防止重复绘图

    n = len(y1)
    x = list(range(1, n + 1))  # 生成长度为n的序列，例如：[1, 2, 3, ..., n]

    plt.scatter(x, y1, label='Predicted Value')
    plt.scatter(x, y2, label='Real Value')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.grid(True)

    # 在图中显示MSE和R2
    plt.text(0.8, 0.9, f'MSE: {rmse:.2f}', transform=plt.gca().transAxes)
    plt.text(0.8, 0.85, f'R2: {r2:.2f}', transform=plt.gca().transAxes)

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # 设置中文字体为黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    plt.legend(loc='upper left')

    # save_directory = './results/MLP'
    os.makedirs(save_directory, exist_ok=True)  # 如果目录不存在，创建目录
    save_path = os.path.join(save_directory, f'{title}.png')
    plt.savefig(save_path, dpi=300)
