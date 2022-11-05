import argparse

from models.ConvNet import ConvNet
from models.ConvNetLarge import ConvNetLarge

# This file contains code to help with the enforcement and parsing of parameters
# provided to the training and infer functions.

def get_parser(caller):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        choices=('basic', 'large'))

    parser.add_argument('--data-dir', type=str, default="./data/")

    parser.add_argument('--model-parameters', type=dict, default={})

    parser.add_argument('--verbose', type=int, default=1)


    if caller == "train":

        parser.add_argument('--max-epoch', type=int, default=10)
        parser.add_argument('--batch-size', type=int, default=128)
        parser.add_argument('--val-split-size', type=float, default=0.2)
        parser.add_argument('--lr', type=float, default=1e-3)

    elif caller == "infer":

        parser.add_argument('--model-filepath', type=str, default="./ConvNet_best_model.pth")

    return parser


def get_arguments(caller):
    parser = get_parser(caller)
    args = parser.parse_args()
    return args

def get_model_class(model_type):

    MODELS = {
        'basic': ConvNet,
        'large': ConvNetLarge
    }

    assert model_type in MODELS
    return MODELS[model_type]
