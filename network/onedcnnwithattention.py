import torch
import torch.nn as nn
import torch.optim as optim

# 自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        self.scale = attention_dim ** 0.5  # 缩放因子
        
    def forward(self, x):
        # x 的形状为 (batch_size, seq_len, input_dim)
        queries = self.query(x)  # (batch_size, seq_len, attention_dim)
        keys = self.key(x)  # (batch_size, seq_len, attention_dim)
        values = self.value(x)  # (batch_size, seq_len, attention_dim)
        
        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale  # (batch_size, seq_len, seq_len)
        attention_weights = torch.softmax(attention_scores, dim=-1)  # (batch_size, seq_len, seq_len)
        
        # 计算加权后的输出
        attention_output = torch.matmul(attention_weights, values)  # (batch_size, seq_len, attention_dim)
        return attention_output

# 卷积神经网络 + 自注意力机制
class CNNWithAttention(nn.Module):
    def __init__(self):
        super(CNNWithAttention, self).__init__()
        # 一维卷积层
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 自注意力层
        self.attention = SelfAttention(input_dim=128, attention_dim=128)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 200, 128)  # 假设池化后光谱长度为200
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)  # 回归输出
        
    def forward(self, x):
        # 卷积层
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)  # 形状为 (batch_size, 64, seq_len/2)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)  # 形状为 (batch_size, 128, seq_len/4)
        
        # 转换为 (batch_size, seq_len, feature_dim) 供注意力机制使用
        x = x.permute(0, 2, 1)  # (batch_size, seq_len, 128)
        
        # 自注意力层
        x = self.attention(x)  # (batch_size, seq_len, 128)
        
        # 展平并传入全连接层
        x = x.view(x.size(0), -1)  # (batch_size, seq_len * 128)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output


