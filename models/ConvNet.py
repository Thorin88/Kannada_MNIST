import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

class ConvNet(nn.Module):

    def __init__(self, in_channels=1, num_classes=10):
        super(ConvNet, self).__init__()

        self.num_classes = num_classes

        self.name = "ConvNet"

        layer1_kernel_num = 16
        self.layer1 = nn.Sequential(

            # Kernel details
            nn.Conv2d(in_channels, layer1_kernel_num, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(layer1_kernel_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        layer2_kernel_num = 32
        self.layer2 = nn.Sequential(
            nn.Conv2d(layer1_kernel_num, layer2_kernel_num, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(layer2_kernel_num),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(7*7*layer2_kernel_num, num_classes)

        self.output_layer = nn.Softmax(dim=1)

        print("Model initialised.")

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        # Flatten output of Conv part
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.output_layer(out)
        return out
