# run the trainers
import os
import torchvision.transforms as transforms
from trainers import Trainer, TrainerParams
from autoencoder2D import ConvAutoencoder
from torch.utils.data import Subset
from torchvision import datasets
from diversityScore import DiversityScore


def main():
    # convert data to torch.FloatTensor
    transform = transforms.ToTensor()

    # load a subset of indices corresponding to images labelled with 7 only
    import pickle as pkl
    f = open("subset_index.pkl", "rb")
    idx = pkl.load(f)
    f.close()

    # load the training and test datasets
    train_data = Subset(datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True, download=True, transform=transform), idx)
    test_data = Subset(datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False, download=True, transform=transform), idx)

    model_name = "autoencoderMNIST.pt"
    model = ConvAutoencoder(save_path=os.path.join("./", model_name))

    params = TrainerParams(n_epochs=5, num_workers=0, batch_size=20)

    trainer = Trainer(model, params, train_data, test_data)

    #trainer.train()
    #trainer.eval()

    output = trainer.get_compressed()

    ds = DiversityScore(output)

    matrix = ds.cosineSimilarity()

    print(matrix.shape)
    print(matrix[0, 0])
    print(matrix[1, 1])

    score = ds.vendiScore()

    print("Vendi score {0:.2f}".format(score))


if __name__ == "__main__":
    main()