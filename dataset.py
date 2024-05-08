from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import os
import numpy as np
import struct
from array import array
import torch


class MNIST(Dataset):
    def __init__(self, root_dir, train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        assert isinstance(train, bool), "Train parameter must be a boolean"
        if train:
            self.images_path = os.path.join(root_dir, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
            self.labels_path = os.path.join(root_dir, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        elif not train:
            self.images_path = os.path.join(root_dir, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
            self.labels_path = os.path.join(root_dir, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

    def __len__(self):
        images, labels = self.read_images_labels()
        return images.shape[0]

    def read_images_labels(self):
        labels = []
        with open(self.labels_path, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(self.images_path, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return np.array(images), np.array(labels)

    # TODO for an autoencoder the label may need to be the same as the image
    def __getitem__(self, idx):
        # load the whole array of images
        images, labels = self.read_images_labels()

        return self.transform(images[idx, :, :]), labels[idx]


def main():
    dataset = MNIST(root_dir='/Users/katecevora/Documents/PhD/data/MNIST', train=True)
    images, labels = dataset.read_images_labels()
    print("Hi")


if __name__ == "__main__":
    main()