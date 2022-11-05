# import albumentations as AUGS

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

from init_helper import get_arguments, get_model_class

from CustomDataset import CustomDataset

# Trains the provided model class using the data provided
# Params:
# - model_class: The model class to be trained. Note: this wants to be given CNN not CNN().
# - model_init_params: A dictionary of parameters and their values to be passed to
# the model class whenever it needs to be reinitialisd.
# - X: The examples data to be split
# - Y: The label data to be split
# - valSplitSize: The size of the validation set, used to split X into training and
# validation sets
# - batch_size: batch size to train the model with
# - num_epochs: The number of epochs to preform during each training phase.
# - lr: The learning rate to use during training
# - verbose: Controls the amount of output printed
# Outputs:
# - The model with the highest validation accuracy seen over the course of training. The
# function also prints where a copy of the weights for this model have been saved to.
def train(model_class,model_init_params,Xtr,Ytr,valSplitSize=0.2,batch_size=32,num_epochs=20,lr=1e-3,verbose=0):

    X, Y = None, None

    trainSize = int((1.0 - valSplitSize)*len(Xtr))
    print("Train set size:",trainSize)
    # Validation set
    Xv, Yv = Xtr[trainSize:], Ytr[trainSize:]
    # Train set
    Xtr, Ytr = Xtr[:trainSize], Ytr[:trainSize]

    print("Training Dataset sizes:","[", Xtr.shape, Ytr.shape,"]")
    print("Validation Dataset sizes:","[", Xv.shape, Yv.shape,"]")

    device = get_device()

    model = model_class(**model_init_params).to(device)
    print("Model Loaded:",model.name)

    # aug_transform = AUGS.Compose([
    #   # Vertical flipping
    #   # AUGS.VerticalFlip(),
    #   # AUGS.HueSaturationValue(p=0.2),
    #   # AUGS.RandomBrightness(limit=0.2)
    #   AUGS.Affine(translate_percent=(0.0, 0.2), p=0.2),
    #   AUGS.Blur(blur_limit=3, always_apply=False, p=0.3),
    #   AUGS.ColorJitter(p=0.7),
    #   ])

    # Due to personal PC being unable to install high enough versions of required libraries
    aug_transform=None

    # Make datasets
    train_dataset = CustomDataset(Xtr, Ytr, num_classes=model.num_classes, aug_transform=aug_transform)
    val_dataset = CustomDataset(Xv, Yv, num_classes=model.num_classes)

    # Make loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    criterion = nn.CrossEntropyLoss()
    print("Loss Function:",criterion)

    # Uses our learning rate parameter
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    results = dict()
    for phase in ["training","validation"]:
        results[phase] = {"loss": [], "accuracy": []}

    lowestValidationAcc = np.NINF

    # Train the model
    for epoch in range(num_epochs):

        results_this_epoch = dict()
        for phase in ["training","validation"]:
            results_this_epoch[phase] = {"loss": [], "accuracy": []}

        # Needed since we switch to .eval() during the validation stage
        model.train()
        for i, (images, labels) in enumerate(tqdm(train_loader)):

            images = images.to(device)
            labels = labels.to(device)
            # Forward pass
            preds = model(images)

            loss = criterion(preds, labels)
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            results_this_epoch["training"]["loss"].append(loss.item())
            results_this_epoch["training"]["accuracy"].append(count_correct(preds, labels))

        # Validation

        # Making sure we aren't using dropouts and also aren't training the model
        model.eval()
        with torch.no_grad():

            for i, (images, labels) in enumerate(tqdm(val_loader)):

                images = images.to(device)
                labels = labels.to(device)
                # Forward pass
                preds = model(images)

                loss = criterion(preds, labels)
                # No update step

                results_this_epoch["validation"]["loss"].append(loss.item())
                results_this_epoch["validation"]["accuracy"].append(count_correct(preds, labels))

        print('Epoch [{}/{}]'.format(epoch+1, num_epochs))

        # Aggregating the metrics collected during the batches for this epoch
        for phase in results_this_epoch.keys():
            for metric in results_this_epoch[phase].keys():
                if metric == "loss":
                    results[phase][metric].append(np.mean(results_this_epoch[phase][metric]))
                elif metric == "accuracy":
                    if phase == "training":
                        total = len(Ytr)
                    elif phase == "validation":
                        total = len(Yv)
                    results[phase][metric].append(np.sum(results_this_epoch[phase][metric])/total)

                if (verbose > 0):
                    print('[{}] {}: {:.4f}'.format(phase, metric, results[phase][metric][-1]))

        # Early stopping, prevents the model returned being overfit when using
        # too many epochs
        if results["validation"]["accuracy"][-1] > lowestValidationAcc:
            lowestValidationAcc = results["validation"]["accuracy"][-1]
            if verbose > 0:
                print("  **")
            state = {'epoch': epoch + 1,
                     'model_dict': model.state_dict(),
                     'params': model_init_params,
                     'best_loss_on_test': lowestValidationAcc}

            modelLocation = model.name + "_best_model.pth"
            torch.save(state, modelLocation) # saves to current directory
            if verbose > 0:
                  print("Model information (including weights) saved to:",modelLocation)

    plotResults(results)

    # Load up saved model
    model = model_class(**model_init_params).to(device)
    model.load_state_dict(torch.load(modelLocation,map_location=torch.device(device))["model_dict"])
    model.eval()

    return model

def main():

    args = get_arguments("train")

    data_dir = args.data_dir

    X, Y = load_data(data_dir + "train.csv")

    trained_model = train(get_model_class(args.model),
                          args.model_parameters,
                          X[:1000],Y[:1000],
                          verbose=args.verbose,
                          num_epochs=args.max_epoch,
                          batch_size=args.batch_size,
                          lr=args.lr,
                          valSplitSize=args.val_split_size)

if __name__ == '__main__':
    main()
