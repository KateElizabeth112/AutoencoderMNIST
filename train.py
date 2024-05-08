# run the trainers
import os
from torchvision import datasets
import torchvision.transforms as transforms
from trainers import Trainer, TrainerParams
from autoencoder2D import ConvAutoencoder


def main():
    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # load the training and test datasets
    train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False, download=True, transform=transform)

    model_name = "autoencoderMNIST.pt"
    model = ConvAutoencoder(save_path=os.path.join("./", model_name))

    params = TrainerParams(n_epochs=5, num_workers=0, batch_size=20)

    trainer = Trainer(model, params, train_data, test_data)

    trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()