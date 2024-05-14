# run the trainers
import mlflow
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from trainers import Trainer, TrainerParams
from autoencoder2D import ConvAutoencoder
from torch.utils.data import Subset
from torchvision import datasets
from diversityScore import DiversityScore
import random

dataset_root = '~/.pytorch/MNIST_data/'

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Set our tracking server uri for logging with MLFlow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("Vendi Score MNIST")


def generateSubsetIndex(data, category, n_samples, random_seed, train=True):
    # generate an index of data samples to use
    assert isinstance(category, int), "The category must be an integer in the range 0-9"
    assert category <= 9, "The category value cannot be greater than 9."
    assert category >= 0, "The category value cannot be less than 0"
    assert isinstance(n_samples, int), "The number of samples must be an integer"

    # TODO why doesn't this work here but it's fine in the main function
    # open the full dataset
    #data = datasets.MNIST(root=dataset_root, train=train, download=False, transform=transforms.ToTensor()),

    # create a data loader
    dataset_loader = torch.utils.data.DataLoader(data, batch_size=20, num_workers=0)

    # iterate over the dataset to get the labels
    labels = []
    for data in dataset_loader:
        _, label = data
        labels += list(label.numpy())

    # convert to numpy array
    labels = np.array(labels)

    # get an index of locations where the label is equal to the value of category
    idx = np.where(labels == category)[0]

    # check that the number of samples is less than the number of datapoints in that category
    assert n_samples <= idx.shape[
        0], "The number of samples ({}) must be less than the number of datapoints in the category ({})".format(
        n_samples, idx.shape[0])

    # sample with a random seed
    random.seed(random_seed)
    random_idx = random.sample(range(0, idx.shape[0]), n_samples)

    subset_idx = idx[random_idx]

    assert subset_idx.shape[0] == n_samples, "The number of samples in the idx_sample array does not match n_samples"

    return subset_idx


def main():
    # Define the model parameters for mlflow
    params = {
        "data_category": 5,
        "n_samples": 500,
        "random_seed": 112,
        "n_layers": 3,
        "n_epochs": 20,
        "n_workers": 0,
        "batch_size": 20,
        "model_name": "autoencoderMNIST.pt"
    }

    # load the training and test datasets
    train_data = datasets.MNIST(root=dataset_root, train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root=dataset_root, train=False, download=False, transform=transform)

    # generate a subset of indices corresponding to images labelled with a given category
    idx_train = generateSubsetIndex(train_data, params["data_category"], params["n_samples"], params["random_seed"], train=True)

    # select a subset of datapoints
    train_data = Subset(datasets.MNIST(root=dataset_root, train=True, download=False, transform=transform), idx_train)

    model = ConvAutoencoder(save_path=os.path.join("./", params["model_name"]))

    trainer_params = TrainerParams(n_epochs=params["n_epochs"], num_workers=params["n_workers"], batch_size=params["batch_size"])

    trainer = Trainer(model, trainer_params, train_data, test_data)

    # trainer.train()
    #trainer.eval(train=True)

    ds = DiversityScore(model, trainer_params, train_data)

    score = ds.vendiScore()
    pixel_vs = ds.vendiScorePixel()

    print("Vendi score {0:.2f}".format(score))
    print("Pixel vendi score {0:.2f}".format(pixel_vs))

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the vs metric
        mlflow.log_metric("vs_encoded", score)
        mlflow.log_metric("vs_pixel", pixel_vs)


if __name__ == "__main__":
    main()
