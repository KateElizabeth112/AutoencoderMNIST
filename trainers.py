# class for training a model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.dataset import Dataset

class TrainerParams:
    def __init__(self, n_epochs=30, num_workers=0, batch_size=20):
        self.n_epochs = n_epochs
        self.num_workers = num_workers
        self.batch_size = batch_size

class Trainer:
    def __init__(self, model, params, train_data, test_data):
        # check that the objects are instances of the correct class
        assert isinstance(params, TrainerParams), "params is not an instance of trainerParams"
        assert isinstance(model, nn.Module), "model is not an instance of nn.Module"
        assert isinstance(train_data, Dataset), "train_data is not an instance of Dataset"
        assert isinstance(test_data, Dataset), "test_data is not an instance of Dataset"

        self.model = model
        self.params = params
        self.train_data = train_data
        self.test_data = test_data

        # TODO move the dataloaders out and into a dataset class
        # prepare data loaders
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.params.batch_size, num_workers=self.params.num_workers)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.params.batch_size, num_workers=self.params.num_workers)

    def train(self):
        # specify loss function
        criterion = nn.MSELoss()

        # specify loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        for epoch in range(1, self.params.n_epochs + 1):
            # monitor training loss
            train_loss = 0.0

            ###################
            # train the model #
            ###################
            for data in self.train_loader:
                # _ stands in for labels, here
                # no need to flatten images
                images, _ = data
                # clear the gradients of all optimized variables
                optimizer.zero_grad()
                # forward pass: compute predicted outputs by passing inputs to the model
                outputs = self.model(images)
                # calculate the loss
                loss = criterion(outputs, images)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                # perform a single optimization step (parameter update)
                optimizer.step()
                # update running training loss
                train_loss += loss.item() * images.size(0)

            # print avg training statistics
            train_loss = train_loss / len(self.train_loader)
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch,
                train_loss
            ))

            # Checkpoint model
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, self.model.save_path)

    # Load a saved model and run the evaluation data through it
    def eval(self):
        # load the model from the latest checkpoint
        checkpoint = torch.load(self.model.save_path)

        # Load the model state dictionary
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # obtain one batch of test images
        dataiter = iter(self.test_loader)
        images, labels = next(dataiter)

        # get sample outputs
        output = self.model(images)
        # prep images for display
        images = images.numpy()

        # output is resized into a batch of iages
        output = output.view(self.params.batch_size, 1, 28, 28)
        # use detach when it's an output that requires_grad
        output = output.detach().numpy()

        # plot the first ten input images and then reconstructed images
        fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

        # input images on top row, reconstructions on bottom
        for images, row in zip([images, output], axes):
            for img, ax in zip(images, row):
                ax.imshow(np.squeeze(img), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        plt.show()


    def plot(self):
        return 0