import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

class ConvNetLarge(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ConvNetLarge, self).__init__()

        self.num_classes = num_classes

        self.name = "ConvNetLarge"

        # Layer 1 variables:
        layer1_kernel_num = 64
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, layer1_kernel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(layer1_kernel_num),
            nn.LeakyReLU(negative_slope=0.1),
            # nn.MaxPool2d(kernel_size=2, stride=2)
            )

        # Layer 2 variables:
        layer2_kernel_num = 64
        self.layer2 = nn.Sequential(
            nn.Conv2d(layer1_kernel_num, layer2_kernel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(layer2_kernel_num),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2))

        layer3_kernel_num = 128
        self.layer3 = nn.Sequential(
            nn.Conv2d(layer2_kernel_num, layer3_kernel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(layer3_kernel_num),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

        layer4_kernel_num = 128
        self.layer4 = nn.Sequential(
            nn.Conv2d(layer3_kernel_num, layer4_kernel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(layer4_kernel_num),
            nn.LeakyReLU(negative_slope=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2)
            )

#         self.fc1 = nn.Sequential(
#             nn.Linear(1152, 256),
#             nn.Dropout(p=0.2)
#             )

        self.fc1 = nn.Linear(1152, num_classes)

        self.output_layer = nn.Softmax(dim=1)

        print("Model initialised.")

    def forward(self, x):

        out = self.layer1(x)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        # Flatten output of Conv part
        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.output_layer(out)
        return out
