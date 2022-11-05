# Setup

Visit the link: https://www.kaggle.com/competitions/Kannada-MNIST/data, and download the files `train.csv` and `test.csv`. Place these files in the `./data/` directory.

# Training a model

1) Navigate to the root directory of this code

2) Run the command `python train.py basic`

This should begin training a model with the default command line parameters, also printing where
the model weights are saved to, as well as showing graphs of the model's performance during training.

You can use the command `python train.py -h` to see a full list
of options you can alter when training the models. The default command provided in step 2 will
yield results similar to those obtained during the Kaggle competition.

As some further clarification of the less self-explanatory options, `--data-dir` specifies the directory
where the `train.csv` and `test.csv` files are stored. Eg `"./data/"`. You can also choose between training
two model classes, `basic` being a smaller CNN and `large` being a larger CNN.

If training is taking a long time, feel free to use the `--limit` option to lower the amount of training data
used, the models can still achieve decent results even when using a limit of 1500. Eg `--limit 1500`.

# Generating Test Predictions

1) Navigate to the root directory of this code

2) Make sure you have completed running the training command mentioned earlier

3) Run the command `python infer.py basic`

This should result in production of a `outputs/submission.csv` file in the root directory of the code base. Again, the
`-h` option can show you any options available for this command, but this default command will be enough to
reproduce results.
