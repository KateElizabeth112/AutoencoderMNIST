import torchvision.transforms as transforms
import torch
import numpy as np
import os


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split="train", task="pneumoniamnist"):
        assert os.path.exists(data_dir), "The data directory {} does not exist".format(data_dir)
        assert isinstance(split, str), "The argument split must be a string."
        assert split in ["train", "test", "val"], "The argument split must be in the set [train, test, valid]"
        assert isinstance(task, str), "The argument task must be a string"

        allowed_tasks = ["pneumoniamnist"]
        assert task in allowed_tasks, "The argument task must be in the set {}".format(allowed_tasks)

        npz_path = os.path.join(data_dir, "MedMNIST/{}.npz".format(task))
        assert os.path.exists(npz_path)

        data = np.load(npz_path)
        images = data["{}_images".format(split)]
        labels = data["{}_labels".format(split)]

        self.images = images
        self.labels = labels
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5], std=[.5])
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.data_transform(self.images[index]), self.labels[index][0]
