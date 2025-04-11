# Description: ELM模型

import torch
import torch.nn as nn

class ELM(nn.Module):
    def __init__(self, feature, hidden):
        super(ELM, self).__init__()
        self.fc1 = nn.Linear(feature, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x