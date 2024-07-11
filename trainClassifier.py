# train a classifier to classify MNIST digits
import mlflow
import os
import numpy as np
import torchvision.transforms as transforms
from trainers import Trainer, TrainerParams, Inference
from classifier import ImageClassifier
from torchvision import datasets
from torch.utils.data import Subset
from datasetUtils import RotationTransform
from plotting import MNISTPlotter

url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
md5 = "58c8d27c78d21e728a6bc7b3cc06412e"

dataset_root = '/Users/katecevora/Documents/PhD/data'

# convert data to torch.FloatTensor
transform = transforms.ToTensor()
transform_emnist = transforms.Compose([transforms.ToTensor(), RotationTransform(-270)])


# generate a unique ID for the model
# unique_id = ''.join(random.choices('0123456789', k=6))
unique_id = "999999"
# generate  a params file
params = {
    "data_category": all,
    "n_samples": all,
    "random_seed": 112,
    "n_layers": 3,
    "n_epochs": 100,
    "n_workers": 0,
    "batch_size": 20,
    "model_name": "classifierMNIST_{}.pt".format(unique_id)
}

subset_size = 10000
train_emnist = Subset(
    datasets.EMNIST(root=dataset_root, split="digits", train=True, download=False, transform=transform_emnist),
    np.arange(0, subset_size))
test_emnist = Subset(
    datasets.EMNIST(root=dataset_root, split="digits", train=False, download=False, transform=transform_emnist),
    np.arange(0, subset_size))

# run with test data from a different dataset
train_mnist = Subset(datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True, download=False, transform=transform),
                     np.arange(0, subset_size))
test_mnist = Subset(datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False, download=False, transform=transform),
                    np.arange(0, subset_size))


def train():
    model = ImageClassifier(save_path=os.path.join("./models", params["model_name"]))

    trainer_params = TrainerParams(n_epochs=params["n_epochs"], num_workers=params["n_workers"],
                                   batch_size=params["batch_size"])

    trainer = Trainer(model, trainer_params, train_emnist, test_emnist, model_type="classifier")

    epochs_record, train_loss_record, test_loss_record = trainer.train()


def runInference():
    model = ImageClassifier(save_path=os.path.join("./models", params["model_name"]))

    inferencer = Inference(model, train_emnist, model_type="classifier")

    accuracy = inferencer.eval()

    print("Self train accuracy is {0:.2f}".format(accuracy))

    # run with test data from the same dataset
    inferencer = Inference(model, test_emnist, model_type="classifier")

    accuracy = inferencer.eval()

    print("Self test accuracy is {0:.2f}".format(accuracy))

    inferencer = Inference(model, train_mnist, model_type="classifier")

    accuracy = inferencer.eval()

    print("Cross train accuracy is {0:.2f}".format(accuracy))

    inferencer = Inference(model, test_mnist, model_type="classifier")

    accuracy = inferencer.eval()

    print("Cross test accuracy is {0:.2f}".format(accuracy))




def main():
    #train()
    #runInference()

    # run with test data from a different dataset
    train_mnist = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True, download=False, transform=transform)

    # plotter = MNISTPlotter(train_mnist, save_path="/Users/katecevora/Documents/PhD/data/MNIST/images", dataset_name="MNIST")
    # plotter.plotAllDigits()


if __name__ == "__main__":
    main()
