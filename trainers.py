# class for training a model
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.utils.data.dataset import Dataset
from plotting import LossPlotter
import os


class TrainerParams:
    def __init__(self, n_epochs=30, num_workers=0, batch_size=20):
        self.n_epochs = n_epochs
        self.num_workers = num_workers
        self.batch_size = batch_size


class Trainer:
    """
    Trainer class for an Autoencoder or classifier model. Takes a model, parameters and train and test data.
    """
    def __init__(self, model, params, train_data, test_data, model_type="AE"):
        # check that the objects are instances of the correct class
        assert isinstance(params, TrainerParams), "params is not an instance of trainerParams"
        assert isinstance(model, nn.Module), "model is not an instance of nn.Module"
        assert isinstance(train_data, Dataset), "train_data is not an instance of Dataset"
        assert isinstance(test_data, Dataset), "test_data is not an instance of Dataset"
        assert isinstance(model_type, str), "model_type must be a string"

        # Run some checks on the value of model type
        if model_type not in ("AE", "classifier", "binary_classifier"):
            raise ValueError("The model type {} does not exist".format(model_type))

        # Check if we have a GPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = model.to(self.device)
        self.params = params
        self.train_data = train_data
        self.test_data = test_data
        self.model_type = model_type

        # prepare data loaders
        self.train_loader = torch.utils.data.DataLoader(self.train_data, batch_size=self.params.batch_size,
                                                        num_workers=self.params.num_workers, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=self.params.batch_size,
                                                       num_workers=self.params.num_workers, shuffle=True)

        print("Finished trainer initialisation. Training on {}".format(self.device))

    def train(self):
        """
        Trains a model for a number of epochs specified in self.params.n_epochs. Saves the model when training
        is complete.
        :return:
        """
        # specify loss function
        if self.model_type == "AE":
            loss_function = nn.MSELoss()
        elif self.model_type == "classifier":
            loss_function = nn.CrossEntropyLoss(reduction="mean")
        elif self.model_type == "binary_classifier":
            loss_function = nn.BCELoss(reduction="mean")
        else:
            raise Exception

        # specify loss function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        # create a container to store the train loss and associated epoch
        train_loss_record = []
        test_loss_record = []
        epochs_record = []

        # Set frequency to save loss value and model
        assert self.params.n_epochs > 0, "Number of epochs cannot be less than or equal to 0."
        assert isinstance(self.params.n_epochs, int), "Number of epochs must be an integer"

        if self.params.n_epochs >= 10000:
            n_steps = 100
        if self.params.n_epochs < 10000:
            n_steps = 20
        if self.params.n_epochs < 1000:
            n_steps = 10
        if self.params.n_epochs < 200:
            n_steps = 5
        if self.params.n_epochs < 100:
            n_steps = 1

        for epoch in range(1, self.params.n_epochs + 1):
            # monitor training loss
            total_train_loss = 0.0

            ###################
            # train the model #
            ###################
            for images, labels in self.train_loader:
                # Make sure the images and labels are on the right type of device (cuda or cpu)
                images = images.to(self.device)
                labels = labels.to(self.device)

                # clear the gradients of all optimized variables
                optimizer.zero_grad()

                # forward pass: compute predicted outputs by passing inputs to the model
                preds, images_compressed = self.model(images)

                # calculate the loss using only the first output from the network
                # loss_function returns average loss across batch
                if self.model_type == "AE":
                    loss = loss_function(preds, images)
                elif self.model_type in ["classifier", "binary_classifier"]:
                    loss = loss_function(preds, labels)

                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                # perform a single optimization step (parameter update)
                optimizer.step()

                # update running training loss
                # Scale it by the size of the batch since loss_function returned batch average
                total_train_loss += loss.item() * images.size(0)

            # print and save training/test statistics every n_steps
            if epoch % n_steps == 0:

                # calculate average training loss (divide by the size of the dataset)
                average_train_loss = total_train_loss / len(self.train_loader.dataset)
                print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                    epoch,
                    average_train_loss
                ))

                # Checkpoint model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': average_train_loss,
                }, self.model.save_path)

                # calculate test loss
                total_test_loss = 0.0
                for images, labels in self.test_loader:
                    # Make sure the images and labels are on the right type of device (cuda or cpu)
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # forward pass: compute predicted outputs by passing inputs to the model
                    preds, images_compressed = self.model(images)

                    # calculate the loss using only the first output from the network
                    if self.model_type == "AE":
                        loss = loss_function(preds, images)
                    elif self.model_type == "classifier":
                        loss = loss_function(preds, labels)

                    # update running testloss
                    total_test_loss += loss.item() * images.size(0)

                # Calculate the average test loss
                average_test_loss = total_test_loss / len(self.test_loader.dataset)
                print('Epoch: {} \tTest Loss: {:.6f}'.format(
                    epoch,
                    average_test_loss
                ))

                # store loss and epoch
                epochs_record.append(epoch)
                train_loss_record.append(average_train_loss)
                test_loss_record.append(average_test_loss)

                # plot the loss
                Plotter = LossPlotter(epochs_record, train_loss_record, test_loss_record, save_path="./")
                Plotter.plotLoss()

        return epochs_record, train_loss_record, test_loss_record

    # Load a saved model and run the evaluation data through it
    def eval(self, train=False, save_path=""):
        """
        Loads a saved model from the latest checkpoint, runs a batch of evaluation data through it and plots the
        predictions.
        :return:
        """
        # check that the model path exists
        assert os.path.exists(self.model.save_path), "The model save path {} does not exist".format(self.model.save_path)

        # check that the plot save path exists
        assert isinstance(save_path, str), "The reconstruction image save path must be a string."
        if save_path != "":
            assert os.path.exists(save_path), "Save path {} for reconstruction image does not exist".format(save_path)

        # load the model from the latest checkpoint
        checkpoint = torch.load(self.model.save_path)

        # Load the model state dictionary
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # obtain one batch of test images
        if train:
            dataiter = iter(self.train_loader)
        else:
            dataiter = iter(self.test_loader)
        images, labels = next(dataiter)

        # Make sure the images and labels are on the right type of device (cuda or cpu)
        images = images.to(self.device)
        labels = labels.to(self.device)

        # get sample outputs
        preds, images_compressed = self.model(images)
        # prep images for display
        images = images.numpy()

        # output is resized into a batch of iages
        output = preds.view(self.params.batch_size, 1, 28, 28)
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

        if save_path == "":
            plt.show()
        else:
            plt.savefig(os.path.join(save_path, "reconstruction.png"))


class Inference:
    """
    Inference class for a trained model (autoencoder or classifer).
    """
    def __init__(self, model, test_data, model_type="AE"):
        # check the types of our arguments
        assert isinstance(model_type, str), "model_type must be a string"
        assert isinstance(model, nn.Module), "model is not an instance of nn.Module"
        assert isinstance(test_data, Dataset), "test_data is not an instance of Dataset"

        # Run some checks on the value of model type
        supported_model_types = ["AE", "classifier", "binary_classifier"]
        if model_type not in supported_model_types:
            raise ValueError("The model type {0} is not recognised. Please set to one of {1}".format(model_type, supported_model_types))

        # Check if we have a GPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        self.model = model.to(self.device)
        self.test_data = test_data
        self.model_type = model_type
        self.test_loader = torch.utils.data.DataLoader(self.test_data, batch_size=10, num_workers=0)

        print("Running inference on {}".format(self.device))
        print("Model type is {}".format(self.model_type))

    def eval(self):
        """
        Loads a saved model from the latest checkpoint, runs a batch of evaluation data through it and plots the
        predictions.
        :return:
        """
        # check that the model path exists
        assert os.path.exists(self.model.save_path), "The model save path {} does not exist".format(self.model.save_path)

        # load the model from the latest checkpoint
        checkpoint = torch.load(self.model.save_path)

        # Load the model state dictionary
        self.model.load_state_dict(checkpoint['model_state_dict'])

        preds_all = []
        labels_all = []

        # if this is a binary classifier, we want the ROC, so store probabilities
        if self.model_type == "binary_classifier":
            probs_all = []

        for images, labels in self.test_loader:
            # Make sure the images and labels are on the right type of device (cuda or cpu)
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits, _ = self.model(images)

            preds = torch.argmax(logits, dim=1)

            # convert to numpy arrays
            preds = preds.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()

            # stack as a list
            preds_all += list(preds)
            labels_all += list(labels)

            # if this is a binary classifier, we want the ROC, so store logits
            if self.model_type == "binary_classifier":
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                probs_all += list(probs[:, 0])

        # calculate accuracy
        error = np.array(preds_all) - np.array(labels_all)
        error[error != 0] = 1
        accuracy = 1 - (np.sum(error) / error.shape[0])

        # if this is a binary classifier, calculate ROC
        if self.model_type == "binary_classifier":
            auroc = roc_auc_score(np.array(labels_all), np.array(probs_all))
            print("AUROC {}".format(auroc))

        return accuracy