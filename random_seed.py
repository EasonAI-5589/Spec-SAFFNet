import torch
import torch.nn as nn
import torch.optim as optim

class Multilayer_Perceptron(torch.nn.Module):
    """
    具有两个隐藏层的前馈神经网络MLP,用于回归预测
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(Multilayer_Perceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()
        # self.sigmoid = torch.nn.Sigmoid() # 非分类问题，不需要sigmoid
        # self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self,x):        
        x = self.relu(self.fc1(x))
        hidden_layer2 = self.relu(self.fc2(x))
        x = self.fc3(hidden_layer2)
        return x, hidden_layer2

# 输出形状是[batchsize,outsize]
    