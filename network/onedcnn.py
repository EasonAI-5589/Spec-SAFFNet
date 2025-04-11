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
        queries = self.query(x)  # (batch_size, seq_len, attention_dim)
        keys = self.key(x)  # (batch_size, seq_len, attention_dim)
        values = self.value(x)  # (batch_size, seq_len, attention_dim)
        
        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        
        # 计算加权后的输出
        attention_output = torch.matmul(attention_weights, values)
        return attention_output

# 基本的1D卷积神经网络
class Basic1DCNN(nn.Module):
    def __init__(self):
        super(Basic1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 200, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # 权重初始化
        self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)
        
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output

    # 初始化权重的方法
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

# 1D卷积神经网络 + 自注意力机制
class CNNWithAttention(nn.Module):
    def __init__(self):
        super(CNNWithAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        # 自注意力层
        self.attention = SelfAttention(input_dim=200, attention_dim=200)

        # Dropout层
        self.dropout = nn.Dropout(0.5)

        # 全连接层
        self.fc1 = nn.Linear(200 * 128, 128)
        self.fc2 = nn.Linear(128, 1)
        
        # 初始化权重
        self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        # 自注意力层
        x = self.attention(x)

        # 展平并传入全连接层
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        output = self.fc2(x)
        return output
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')  # Kaiming 初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.constant_(m.bias, 0)



# 损失函数和优化器定义函数
def get_criterion_and_optimizer(model, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return criterion, optimizer

# Early Stopping 类
class EarlyStopping:
    def __init__(self, patience=5, threshold=1, verbose=False):
        self.patience = patience
        self.threshold = threshold
        self.counter = 0
        self.early_stop = False
        self.verbose = verbose
    
    def __call__(self, loss):
        if loss < self.threshold:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.counter = 0
