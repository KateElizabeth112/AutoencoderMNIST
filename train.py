# run the trainers
import os
import torchvision.transforms as transforms
from trainers import Trainer, TrainerParams
from autoencoder2D import ConvAutoencoder
from dataset import MNIST


def main():
    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # load the training and test datasets
    train_data = MNIST(root_dir='/Users/katecevora/Documents/PhD/data/MNIST', train=True, transform=transform)
    test_data = MNIST(root_dir='/Users/katecevora/Documents/PhD/data/MNIST', train=False, transform=transform)

    model_name = "autoencoderMNIST.pt"
    model = ConvAutoencoder(save_path=os.path.join("./", model_name))

    params = TrainerParams(n_epochs=5, num_workers=0, batch_size=20)

    trainer = Trainer(model, params, train_data, test_data)

    #trainer.train()
    trainer.eval()


if __name__ == "__main__":
    main()