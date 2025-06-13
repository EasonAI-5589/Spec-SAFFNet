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
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # 全连接层
        self.fc1 = nn.Linear(16 * 200, 512)# 输入是800 * 1，卷积核是3，padding是1，所以输出是800 * 8，池化后是400 * 8
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 1)
        
        # 权重初始化
        self._initialize_weights()
        
    def forward(self, x):
        x = self.conv1(x) # (batch_size, 8, 800)
        x = torch.relu(x) # 激活函数
        x = self.pool1(x) # (batch_size, 8, 400)
        
        x = self.conv2(x) # (batch_size, 16, 400)
        x = torch.relu(x) # 激活函数
        x = self.pool2(x) # (batch_size, 16, 200)
        
        x = x.view(x.size(0), -1) # 展平 (batch_size, 16 * 200)
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

    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)



class CNNWithAttention(nn.Module):
    def __init__(self, input_dim=800, conv1_out_channels=8, conv2_out_channels=16, 
                 kernel_size=3, pool_size=2, dropout_rate=0.5, 
                 fc1_dim=512, fc2_dim=64, output_dim=1):
        super(CNNWithAttention, self).__init__()

        # 卷积层1
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=conv1_out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=pool_size)
        
        # 卷积层2
        self.conv2 = nn.Conv1d(in_channels=conv1_out_channels, out_channels=conv2_out_channels, kernel_size=kernel_size, stride=1, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=pool_size)

        # 自注意力层
        self.attention = SelfAttention(input_dim=input_dim // (pool_size ** 2), attention_dim=input_dim // (pool_size ** 2))

        # Dropout层
        self.dropout = nn.Dropout(dropout_rate)

        # 全连接层
        self.fc1 = nn.Linear(conv2_out_channels * (input_dim // (pool_size ** 2)), fc1_dim)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        self.fc3 = nn.Linear(fc2_dim, output_dim)

        # 初始化权重
        self._initialize_weights()
        
    def forward(self, x):
        # 通过卷积层和池化层
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
        x = torch.relu(self.fc2(x))
        output = self.fc3(x)
        return output
    
    def _initialize_weights(self):
        # Kaiming 初始化
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










import torch
import torch.nn as nn

class Improved1DCNN(nn.Module):
    def __init__(self):
        super(Improved1DCNN, self).__init__()
        
        # 第一层卷积
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # 第二层卷积
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # 第三层卷积（可选）
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # 全局平均池化
        # self.global_avg_pool = nn.AdaptiveAvgPool1d(1)  # 输出长度为1
        
        # 全连接层
        self.fc1 = nn.Linear(256*100, 1024)  # 256 个特征（如果使用第三层卷积）
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 1)  
        
        # Dropout层
        self.dropout = nn.Dropout(p=0.5)
        
        # 权重初始化
        self._initialize_weights()
    
    def forward(self, x):
        x = self.conv1(x) # (batch_size, 64, 800)
        x = self.bn1(x)
        x = torch.relu(x) 
        x = self.pool1(x) # (batch_size, 64, 400)
        
        x = self.conv2(x) # (batch_size, 128, 400)
        x = self.bn2(x)
        x = torch.relu(x) 
        x = self.pool2(x) # (batch_size, 128, 200)
        
        x = self.conv3(x) # (batch_size, 256, 200)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool3(x) # (batch_size, 256, 100)
        
        # 全局平均池化
        # x = self.global_avg_pool(x)  # 形状: (batch_size, 256, 1)
        # x = x.squeeze(-1)            # 形状: (batch_size, 256)
        x = x.view(x.size(0), -1) # 展平 (batch_size, 256 * 100)   
        
        # 全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = self.dropout(x)
        output = self.fc3(x)
        return output

attention = SelfAttention(200, 200)
print(attention)