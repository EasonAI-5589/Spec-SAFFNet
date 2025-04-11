import torch
import torch.nn as nn


class one_dim_CNN(nn.Module):
    def __init__(self, input_size, hidden_size_1 , hidden_size_2, conv_kernel_size, pool_kernel_size):
        super(one_dim_CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, hidden_size_1, conv_kernel_size)
        self.pool = nn.MaxPool1d(pool_kernel_size)
        # 计算卷积和池化后的长度
        conv_length = hidden_size_1*((input_size - conv_kernel_size + 1) // pool_kernel_size)
        self.fc1 = nn.Linear(conv_length, hidden_size_2)
        self.fc2 = nn.Linear(hidden_size_2, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x 