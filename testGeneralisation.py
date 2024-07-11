# Takes a parameters file, runs a generalisation experiment and measures diversity of datastets
import argparse
import mlflow
import os
import torchvision.transforms as transforms
from trainers import Trainer, TrainerParams, Inference
from autoencoder2D import ConvAutoencoder
from classifier import ImageClassifier
from torch.utils.data import Subset
from torchvision import datasets
from diversityScore import DiversityScore
import pickle as pkl
from datasetUtils import generateSubsetIndex, RotationTransform
import plotting
import numpy as np

# Set up the argument parser
parser = argparse.ArgumentParser(description="Calculate the generalisation ability and diversity scores for a dataset")
parser.add_argument("-e", "--experiment", type=str, help="Name of the experiment.", default="GeneralisatonEqualCats")
parser.add_argument("-p", "--params_file", type=str, help="Name of params file.", default="test_params.pkl")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located", default="/Users/katecevora/Documents/PhD")

args = parser.parse_args()

root_dir = args.root_dir
code_dir = os.path.join(root_dir, "code/AutoencoderMNIST")
data_dir = os.path.join(root_dir, "data")
params_file_path = os.path.join(code_dir, "params", args.experiment, args.params_file)
models_path = os.path.join(code_dir, "models")
model_save_path = os.path.join(code_dir, "models")
loss_plot_save_path = os.path.join(code_dir, "loss.png")

dataset_root_mnist = '~/.pytorch/MNIST_data/'
dataset_root_emnist = '/Users/katecevora/Documents/PhD/data'

# number of test/valid dataset samples per category
number_test_samples_per_cat = 500

# convert data to torch.FloatTensor
transform_mnist = transforms.ToTensor()
transform_emnist = transforms.Compose([transforms.ToTensor(), RotationTransform(-270)])

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

    # check what dataset we are using and load the data
    dataset_name = params["dataset_name"]
    assert isinstance(dataset_name, str), "Dataset name must be a string."
    assert dataset_name in ["MNIST", "EMNIST"], "The dataset name {} is not recognised."

    if dataset_name == "MNIST":
        train_data = datasets.MNIST(root=dataset_root_mnist, train=True, download=False, transform=transform_mnist)
        valid_data = datasets.MNIST(root=dataset_root_mnist, train=False, download=False, transform=transform_mnist)
        test_data = datasets.EMNIST(root=dataset_root_emnist, split="digits", train=False, download=False,
                                    transform=transform_emnist)
    else:
        train_data = datasets.EMNIST(root=dataset_root_emnist, split="digits", train=False, download=False,
                                     transform=transform_emnist)
        valid_data = datasets.EMNIST(root=dataset_root_emnist, split="digits", train=True, download=False,
                                     transform=transform_emnist)
        test_data = datasets.MNIST(root=dataset_root_mnist, train=False, download=False, transform=transform_mnist)

    # generate a subset of indices corresponding to the dataset size per category
    idx_train = []
    idx_valid = []
    idx_test = []
    for j in range(10):
        idx_train += list(generateSubsetIndex(train_data, j, params["n_samples"], params["random_seed"]))
        idx_valid += list(generateSubsetIndex(valid_data, j, number_test_samples_per_cat, params["random_seed"]))
        idx_test += list(generateSubsetIndex(test_data, j, number_test_samples_per_cat, params["random_seed"]))

    # convert indices back to numpy
    idx_train = np.array(idx_train)
    idx_valid = np.array(idx_valid)
    idx_test = np.array(idx_test)

    # Take a subset from each dataset specified by the indices
    train_data = Subset(train_data, idx_train)
    valid_data = Subset(valid_data, idx_valid)
    test_data = Subset(test_data, idx_test)

    # load the AE model that we will use to embed the data
    model_ae = ConvAutoencoder(save_path=os.path.join(models_path, "autoencoderMNISTfull_339108.pt"))

    trainer_params = TrainerParams(n_epochs=params["n_epochs"], num_workers=params["n_workers"],
                                   batch_size=params["batch_size"])

    # diversity score all datasets
    ds_train = DiversityScore(train_data, trainer_params, model_ae)
    ds_test = DiversityScore(test_data, trainer_params, model_ae)
    ds_valid = DiversityScore(valid_data, trainer_params, model_ae)

    [vs_pixel_train, vs_embed_full_train, vs_embed_partial_train, vs_inception_train,
     entropy_train] = ds_train.scoreDiversity()
    [vs_pixel_test, vs_embed_full_test, vs_embed_partial_test, vs_inception_test,
     entropy_test] = ds_test.scoreDiversity()
    [vs_pixel_valid, vs_embed_full_valid, vs_embed_partial_valid, vs_inception_valid,
     entropy_valid] = ds_valid.scoreDiversity()

    # train the classifier and test the generalisation accuracy
    model = ImageClassifier(save_path=os.path.join(model_save_path, params["model_name"]))

    trainer_params = TrainerParams(n_epochs=params["n_epochs"], num_workers=params["n_workers"],
                                   batch_size=params["batch_size"])

    trainer = Trainer(model, trainer_params, train_data, valid_data, model_type="classifier")

    _ = trainer.train()

    # run inference on the test and validation data
    inferencer = Inference(model, valid_data, model_type="classifier")
    valid_accuracy = inferencer.eval()

    inferencer = Inference(model, test_data, model_type="classifier")
    test_accuracy = inferencer.eval()

    # record everything in MLFlow
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Log the diversity metrics
        # Pixel Vendi score
        mlflow.log_metric("vs_pixel_train", vs_pixel_train)
        mlflow.log_metric("vs_pixel_valid", vs_pixel_valid)
        mlflow.log_metric("vs_pixel_test", vs_pixel_test)

        # Vendi score with embeddings from full dataset
        mlflow.log_metric("vs_embed_full_train", vs_embed_full_train)
        mlflow.log_metric("vs_embed_full_valid", vs_embed_full_valid)
        mlflow.log_metric("vs_embed_full_test", vs_embed_full_test)

        # Vendi score with embeddings from partial dataset
        mlflow.log_metric("vs_embed_partial_train", vs_embed_partial_train)
        mlflow.log_metric("vs_embed_partial_valid", vs_embed_partial_valid)
        mlflow.log_metric("vs_embed_partial_test", vs_embed_partial_test)

        # Vendi score with embeddings from inception
        mlflow.log_metric("vs_inception_train", vs_inception_train)
        mlflow.log_metric("vs_inception_valid", vs_inception_valid)
        mlflow.log_metric("vs_inception_test", vs_inception_test)

        # Label entropy diversity score
        mlflow.log_metric("vs_entropy_train", entropy_train)
        mlflow.log_metric("vs_entropy_valid", entropy_valid)
        mlflow.log_metric("vs_entropy_test", entropy_test)

        # log the generalisation accuracy
        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("valid_accuracy", valid_accuracy)

        # log the loss plot for the classification model
        mlflow.log_artifact(loss_plot_save_path)


if __name__ == "__main__":
    main()
