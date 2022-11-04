
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from torch.autograd import Variable

def load_data(filename, labels_present=True):

    data = pd.read_csv(filename)

    if labels_present:

        X = data.iloc[:,1:].to_numpy().astype("float32").reshape(-1,28,28)/255.0
        print("Training images shape:", X.shape)

        Y = data["label"].to_numpy().astype("float32").reshape(-1,)
        print("Training labels shape:", Y.shape)

        return X, Y

    else:

        X = data.iloc[:,1:].to_numpy().astype("float32").reshape(-1,28,28)/255.0
        print("Test images shape:", X.shape)

        ids = data["id"].to_numpy().astype("int").reshape(-1,)
        print("IDs shape:", ids.shape)

        return X, ids

def get_device():
    if torch.cuda.is_available():
        return torch.device(f'cuda')
    else:
        return torch.device(f'cpu')

def cuda(v):
    if torch.cuda.is_available():
        return v.cuda()
    return v

def toTensor(v,dtype = torch.float,requires_grad = False):
    return cuda(Variable(torch.tensor(v)).type(dtype).requires_grad_(requires_grad))

def toNumpy(v):
    if torch.cuda.is_available():
        return v.detach().cpu().numpy()
    return v.detach().numpy()

# Code which goes through the results collected during training and plots them.
def plotResults(trainingResults):

    metrics_available = trainingResults[list(trainingResults.keys())[0]].keys()

    for i, metric in enumerate(metrics_available):

        plt.figure(figsize=(10,5))

        plt.title(metric + " over the course of training")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        # Values were recorded for train and validation results
        for phase in trainingResults.keys():

            epochs = len(trainingResults[phase][metric])
            epochs_range = range(1,epochs+1)

            plt.plot(epochs_range,trainingResults[phase][metric],label=phase)

        plt.legend()
        plt.show()

def count_correct(preds, labels, debug=False):

    correct_count = 0.0

    predicted_labels = torch.argmax(preds, dim=1)
    correct_count += (predicted_labels == labels).sum().item()

    if debug:
        print("Predicted:",toNumpy(predicted_labels))
        print("Labels:",toNumpy(labels))
        print("Correct Count:",correct_count)

    return correct_count
