# train a classifier to classify MNIST digits
import mlflow
import os
import numpy as np
import torchvision.transforms as transforms
from trainers import Trainer, TrainerParams
from classifier import ImageClassifier
from torchvision import datasets
from torch.utils.data import Subset
import random

url = 'https://biometrics.nist.gov/cs_links/EMNIST/gzip.zip'
md5 = "58c8d27c78d21e728a6bc7b3cc06412e"


dataset_root = '/Users/katecevora/Documents/PhD/data'

# convert data to torch.FloatTensor
transform = transforms.ToTensor()


def main():
    train_data = Subset(datasets.EMNIST(root=dataset_root, split="digits", train=True, download=False, transform=transform), np.arange(0, 1000))
    test_data = datasets.EMNIST(root=dataset_root, split="digits", train=False, download=False, transform=transform)

    # generate a unique ID for the model
    #unique_id = ''.join(random.choices('0123456789', k=6))
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

    model = ImageClassifier(save_path=os.path.join("./models", params["model_name"]))

    trainer_params = TrainerParams(n_epochs=params["n_epochs"], num_workers=params["n_workers"], batch_size=params["batch_size"])

    trainer = Trainer(model, trainer_params, train_data, test_data, model_type="classifier")

    train_epochs, train_loss = trainer.train()


    print("Done")


if __name__ == "__main__":
    main()

