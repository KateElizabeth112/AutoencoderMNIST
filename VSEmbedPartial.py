# Script to calculate VS for a dataset which is encoded by an AE that is trained only on that dataset
# Calculate the Vendi score for a dataset from embeddings of an AE model
import argparse
import mlflow
import os
import torchvision.transforms as transforms
from trainers import Trainer, TrainerParams
from autoencoder2D import ConvAutoencoder
from torch.utils.data import Subset
from torchvision import datasets
from diversityScore import DiversityScore
import pickle as pkl
from datasetUtils import generateSubsetIndex
import plotting

# Set up the argument parser
parser = argparse.ArgumentParser(description="Calculate the Vendi score for a dataset from embeddings of an AE model")
parser.add_argument("-e", "--experiment", type=str, help="Name of the experiment.", default="MNIST_Embed_VS_Partial")
parser.add_argument("-p", "--params_file", type=str, help="Name of params file.", default="test_params.pkl")

args = parser.parse_args()

params_file_path = os.path.join("./params", args.experiment, args.params_file)

dataset_root = '~/.pytorch/MNIST_data/'

# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# Set our tracking server uri for logging with MLFlow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Create a new MLflow Experiment
mlflow.set_experiment(args.experiment)


def main():
    # open a params file
    assert os.path.exists(params_file_path), "The path {} to the params file does not exist.".format(params_file_path)

    f = open(params_file_path, "rb")
    params = pkl.load(f)
    f.close()

    # check the params file
    assert isinstance(params, dict), f"Expected a dictionary, but got {type(params).__name__}"

    # load the training data
    train_data = datasets.MNIST(root=dataset_root, train=True, download=False, transform=transform)
    test_data = datasets.MNIST(root=dataset_root, train=False, download=False, transform=transform)

    # generate a subset of indices corresponding to images labelled with a given category
    idx_train = generateSubsetIndex(train_data, params["data_category"], params["n_samples"], params["random_seed"], train=True)

    # select a subset of datapoints
    train_data = Subset(datasets.MNIST(root=dataset_root, train=True, download=False, transform=transform), idx_train)

    # Initialise a model with the name specified in the params dictionary for this experiment
    model = ConvAutoencoder(save_path=os.path.join("./models", params["model_name"]))

    # Train the model on the dataset
    trainer_params = TrainerParams(n_epochs=params["n_epochs"], num_workers=params["n_workers"],
                                   batch_size=params["batch_size"])
    trainer = Trainer(model, trainer_params, train_data, test_data)
    train_epochs, train_loss = trainer.train()

    # Plot the loss
    save_path = "./"
    Plotter = plotting.LossPlotter(train_epochs, train_loss, save_path=save_path)
    Plotter.plotTrainLoss()

    # Calculate the Vendi score using the embeddings from the model we just trained
    ds = DiversityScore(model, trainer_params, train_data)

    vs_encode = ds.vendiScore()

    print("Vendi score {0:.2f}".format(vs_encode))

    # Start an MLflow run
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the vs metric
        mlflow.log_metric("vs_encoded", vs_encode)

        # log the loss plot
        mlflow.log_artifact(os.path.join(save_path, "loss.png"))



if __name__ == "__main__":
    main()
