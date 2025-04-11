# Description: MLP模型

import torch
import torch.nn as nn


class MLP(torch.nn.Module):
    def __init__(self, feature, hidden1, hidden2):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(feature, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)

    def forward(self,x):        
        x = self.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MLP_3_hidden(torch.nn.Module):
    def __init__(self, feature, hidden1, hidden2, hidden3):
        super(MLP_3_hidden, self).__init__()
        self.fc1 = nn.Linear(feature, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.residual = nn.Linear(feature, 1)

    def forward(self,x):        
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class MLP_3_hidden_with_Residual(torch.nn.Module):
    def __init__(self, feature, hidden1, hidden2, hidden3):
        super(MLP_3_hidden_with_Residual, self).__init__()
        self.fc1 = nn.Linear(feature, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.fc4 = nn.Linear(hidden3, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        
        # 保证输入和输出的维度相同
        self.residual = nn.Linear(feature, 1)

    def forward(self, x):
        identity = self.residual(x)  # 输入作为残差
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        x += identity  # 添加残差
        return x
