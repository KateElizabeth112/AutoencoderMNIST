# train an Autoencoder on the full MNIST dataset and save the model for inference
import mlflow
import os
import numpy as np
import torchvision.transforms as transforms
from trainers import Trainer, TrainerParams
from autoencoder2D import ConvAutoencoder
from torchvision import datasets
import matplotlib.pyplot as plt
import random


dataset_root = '~/.pytorch/MNIST_data/'

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Set our tracking server uri for logging with MLFlow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment("trainAEFull")


def main():
    # generate a unique ID for the model
    unique_id = ''.join(random.choices('0123456789', k=6))
    # generate  a params file
    params = {
        "data_category": all,
        "n_samples": all,
        "random_seed": 112,
        "n_layers": 3,
        "n_epochs": 100,
        "n_workers": 0,
        "batch_size": 20,
        "model_name": "autoencoderMNISTfull_{}.pt".format(unique_id)
    }

    # load the training and test datasets
    train_data = datasets.MNIST(root=dataset_root, train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root=dataset_root, train=False, download=False, transform=transform)

    model = ConvAutoencoder(save_path=os.path.join("./models", params["model_name"]))

    trainer_params = TrainerParams(n_epochs=params["n_epochs"], num_workers=params["n_workers"], batch_size=params["batch_size"])

    trainer = Trainer(model, trainer_params, train_data, test_data)

    train_epochs, train_loss = trainer.train()

    # plot
    plt.plot(train_epochs, train_loss)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("loss.png")

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        mlflow.log_artifact("loss.png")



if __name__ == "__main__":
    main()
