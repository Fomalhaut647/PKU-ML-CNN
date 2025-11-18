import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, num_classes=200):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=2)
        self.pool = nn.MaxPool2d(4, 4)
        self.fc = nn.Linear(16 * 28 * 28, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv(x)))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
