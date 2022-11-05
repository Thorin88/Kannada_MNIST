import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms

from tqdm import tqdm

# Package imports
from models.ConvNet import ConvNet
from models.ConvNetLarge import ConvNetLarge

from helpers import load_data, get_device, cuda, toTensor, toNumpy, plotResults, count_correct

from CustomDataset import CustomDataset

# Using the trained model provided, loads in raw data from the filepath provided
# and returns the model's predictions for this data.
# Inputs:
# - filename: The filepath to the data file
# - model: A pretrained model
# Outputs:
# A dataframe who's first column is the id of the example, and second column is
# the predicted label. The line 'output_df.to_csv("./submission.csv", index=False)'
# will result in the correct output file being generated for the Kaggle competition.
def infer_raw(filename, model):

    device = get_device()

    X, ids = load_data(filename, labels_present=False)

    dataset = CustomDataset(X, np.zeros((len(X),)), num_classes=model.num_classes)

    X = None # Free X

    # Make loaders
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    predicted_labels = np.asarray([],dtype="int")

    model.eval()
    with torch.no_grad():

        for i, (images, _) in enumerate(tqdm(data_loader)):

            images = images.to(device)
            # Forward pass
            preds = model(images)

            predicted_labels = np.concatenate( (predicted_labels, toNumpy(torch.argmax(preds, dim=1))) )

    predicted_labels = predicted_labels.reshape(-1,1)
    ids = ids.reshape(-1,1)

    return pd.DataFrame( np.concatenate( (ids, predicted_labels), axis=1 ), columns = ["id","label"] )

def main():

    device = get_device()

    data_dir = "./data/"

    # TODO - Params for this

    model_init_params = {}
    trained_model = ConvNet(**model_init_params).to(device)
    trained_model.load_state_dict(torch.load("./ConvNet_best_model.pth",map_location=torch.device(device))["model_dict"])
    trained_model.eval()

    results = infer_raw(data_dir + "test.csv", trained_model)
    results.to_csv("./submission.csv", index=False)

if __name__ == '__main__':
    main()
