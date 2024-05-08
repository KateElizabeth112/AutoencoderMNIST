from torch.utils.data.dataset import Dataset
import os
import numpy as np
import struct
from array import array


class MNIST(Dataset):
    def __init__(self, root_dir, train=True, transform=None, subset_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        # if the subset index is not none, check that it is an integer list
        if not (subset_idx is None):
            assert isinstance(subset_idx, (list, np.ndarray)), "subset_idx is not an array or list"
            assert isinstance(subset_idx[0], np.int64), "idx array element is not an np.int64 integer"
        self.subset_idx = subset_idx

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
        import pickle as pkl
        if self.train:
            f = open(os.path.join(self.root_dir, "train_data.pkl"), "rb")
            [images, labels] = pkl.load(f)
            f.close()
        else:
            f = open(os.path.join(self.root_dir, "test_data.pkl"), "rb")
            [images, labels] = pkl.load(f)
            f.close()

        # if we have a subset idx, split the array to include only the listed indices
        if not (self.subset_idx is None):
            images = images[self.subset_idx, :]
            labels = labels[self.subset_idx]

        return images, labels

    # TODO for an autoencoder the label may need to be the same as the image
    def __getitem__(self, idx):
        # load the whole array of images
        images, labels = self.read_images_labels()

        return self.transform(images[idx, :, :]), labels[idx]


def main():
    dataset = MNIST(root_dir='/Users/katecevora/Documents/PhD/data/MNIST', train=False)
    images, labels = dataset.read_images_labels()

    # get an index of locations where the label is equal to 7 only
    idx = np.where(labels == 7)[0]

    # save this index for use later
    import pickle as pkl
    f = open("subset_index.pkl", "wb")
    pkl.dump(idx, f)
    f.close()



if __name__ == "__main__":
    main()