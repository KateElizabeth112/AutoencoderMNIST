import medmnist
from medmnist import INFO
import torchvision.transforms as transforms
import torch


class MedNISTDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels, data_transform, label_transform):
        self.images = images
        self.labels = labels
        self.data_transform = data_transform
        self.label_transform = label_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.data_transform(self.images[index]), self.labels[index]


def getMedNISTData(split="train", task="pneumoniamnist"):
    info = INFO[task]

    assert isinstance(split, str), "The argument split must be a string."
    assert split in ["train", "test", "val"], "The argument split must be in the set [train, test, valid]"
    assert isinstance(task, str), "The argument task must be a string"
    allowed_tasks = ["pneumoniamnist"]
    assert task in allowed_tasks, "The argument task must be in the set {}".format(allowed_tasks)

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    label_transform = transforms.ToTensor()

    # load the data
    DataClass = getattr(medmnist, info['python_class'])
    data = DataClass(split=split, download=True)

    # Create custom dataset objects
    images = []
    labels = []

    for i in range(len(data)):
        x, y = data[i]
        images.append(x)
        labels.append(y[0])

    dataset = MedNISTDataset(images, labels, data_transform, label_transform)

    return dataset