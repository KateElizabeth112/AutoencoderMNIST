# tools to plot the results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset
from scipy.stats import pearsonr

lblu = "#add9f4"
lred = "#f36860"

class ResultsPlotter:
    """
    Class of methods for plotting results - the vendi score calculated for different dataset sizes.
    :param metric: String specifying the metric to plot.
    :param csv_path: String path specifying then results CSV.
    """
    def __init__(self, metric="", experiment_name=""):
        # Check the types of the arguments are as expected
        assert isinstance(experiment_name, str), "The experiment name must be a string"
        assert os.path.exists(os.path.join("./results", experiment_name + ".csv")), "Path to results CSV does not exist."
        assert isinstance(metric, str), "Please specify the metric to be plotted, {}, as a string.".format(metric)

        self.experiment_name = experiment_name
        self.csv_path = os.path.join("./results", experiment_name + ".csv")
        self.metric = metric

        # Run some checks on the value of experimennt
        if self.experiment_name not in ("MNIST_Embed_Inception", "MNIST_Embed_VS", "MNIST_Embed_VS_Full", "MNIST_Pixel_VS"):
            raise ValueError("The experiment {} does not exist".format(self.experiment_name))

        # Check the value of metric
        if self.metric not in ("vs_pixel", "vs_encoded", ""):
            raise ValueError("The metric {} for plotting is not recognised.".format(self.metric))
        if self.metric == "":
            raise ValueError("Please set the metric to be plotted.")


    """
    Method for plotting the given metric for each data category and training set size.
    """
    def plot(self):
        # open the csv
        results = pd.read_csv(self.csv_path)

        # Find out categories our image data takes and order them
        categories = np.sort(np.unique(results["data_category"].values))

        # find out what values n_samples takes and order them
        n_samples = np.sort(np.unique(results["n_samples"].values))

        # initiate plotting
        plt.clf()
        ax = plt.gca()
        # iterate through the categories and plot how the diversity score varies with dataset size
        for cat in categories:
            # filter data belonging to category
            values = []
            for N in n_samples:
                # find the results matching category and number of samples
                results_cat = results[(results["data_category"] == cat) & (results["n_samples"] == N)][self.metric].values

                # Check that we have at least one result for this combo
                assert results_cat.shape[0] != 0, "There are no experiments with category {0} and n_samples {1}".format(cat, N)

                values.append(results_cat[0])

            # plot the series for a category
            ax.plot(n_samples, values, label="{}".format(cat))

        ax.legend(title="Digit Label", loc='upper left', bbox_to_anchor=(1, 1))
        ax.set_title(self.experiment_name)
        ax.set_xticks(n_samples, n_samples)
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Vendi Score")
        ax.set_xscale('log')
        plt.tight_layout()
        plt.show()


class LossPlotter:
    """
    Class of methods for plotting loss curves.

    :param epochs: List of epochs where the loss was recorded
    :param loss: List of loss values averaged over training/test set
    :param save_path: Path where the loss plot will be saved.
    """
    def __init__(self, epochs, train_loss, test_loss, save_path=""):
        # run some checks
        assert isinstance(epochs, list), "The epochs variable must be of type list"
        assert isinstance(train_loss, list), "The train loss variable must be of type list"
        assert isinstance(test_loss, list), "The test loss variable must be of type list"
        assert isinstance(save_path, str), "The save_path variable must be of type str"
        assert os.path.exists(save_path), "The save_path {} does not exist".format(save_path)

        self.epochs = epochs
        self.train_loss = train_loss
        self.test_loss = test_loss
        self.save_path = save_path

    """
    Plot the training loss curve.
    """
    def plotLoss(self):
        plt.plot(self.epochs, self.train_loss, label="Train")
        plt.plot(self.epochs, self.test_loss, label="Test")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        if self.save_path == "":
            plt.show()
        else:
            plt.savefig(os.path.join(self.save_path, "loss.png"))


class MNISTPlotter:
    """
    Class for plotting images from an MNIST-type dataset.
    """
    def __init__(self, dataset, save_path="", dataset_name=""):
        # check the arguments
        assert isinstance(dataset, Dataset), "dataset variable is not of type Dataset"
        assert isinstance(save_path, str), "save_path must be a string"
        assert os.path.exists(save_path), "save_path {} cannot be found".format(save_path)
        assert isinstance(dataset_name, str), "dataset_name must be a string"

        # assign variables
        self.dataset = dataset
        self.save_path = save_path
        self.dataset_name = dataset_name

    def plotDigit(self, digit):
        # check we have a valid digit
        assert isinstance(digit, int), "Digit must be an integer value"
        assert digit in range(0, 10), "Digit must be in the range 0-9"

        # Plot 25 images of a single digit
        labels = self.dataset.targets.numpy()
        index = np.where(labels == digit)[0]

        # create a dataloader for examples with labels corresponding to the value of digit
        dataloader = torch.utils.data.DataLoader(Subset(self.dataset, index), batch_size=5, num_workers=0)
        dataiter = iter(dataloader)

        # plot 25 images
        fig, axes = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)

        for i in range(5):
            images, labels = next(dataiter)
            images = images.numpy()

            for ax, img in zip(axes[i, :], images):
                ax.imshow(np.squeeze(img), cmap='gray')
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

        if self.dataset_name != "":
            plt.title(self.dataset_name)

        if self.save_path == "":
            plt.show()
        else:
            plt.savefig(os.path.join(self.save_path, "digit_{}.png".format(digit)))

    def plotAllDigits(self):
        # plot 25 examples of each digit
        for i in range(10):
            self.plotDigit(i)


class GeneralisationPlotter:
    """
    Class for plotting results of generalisation experiments.
    """
    def __init__(self, experiment_name=""):
        # Check the types of the arguments are as expected
        assert isinstance(experiment_name, str), "The experiment name must be a string"
        assert os.path.exists(os.path.join("./results", experiment_name + ".csv")), "Path to results CSV does not exist."

        self.experiment_name = experiment_name
        self.csv_path = os.path.join("./results", experiment_name + ".csv")

        # Run some checks on the value of experimennt
        if self.experiment_name not in ("Generalisation_Fixed_Entropy", "GeneralisationMinMaxDiversity"):
            raise ValueError("The experiment {} does not exist".format(self.experiment_name))

    def plotGeneralisationGap(self):
        results = pd.read_csv(self.csv_path)

        diversity_scores = ["vs_embed_full_train", "vs_entropy_train", "vs_pixel_train", "vs_inception_train", "n_samples"]
        plot_titles = ["Vendi Score (AE Embedding)", "Label Entropy", "Vendi Score (Raw Pixel)", "Vendi Score (Inception Embedding)", "Number of samples"]

        fig, axes = plt.subplots(nrows=3, ncols=2, sharey=True)

        for c, ds in zip([lred, lblu], ["MNIST", "EMNIST"]):
            valid_accuracy = results["valid_accuracy"][results["dataset_name"] == ds].values
            test_accuracy = results["test_accuracy"][results["dataset_name"] == ds].values

            generalisation_gap = test_accuracy - valid_accuracy / (0.5 * (test_accuracy + valid_accuracy))

            for i in range(5):
                ax = axes.flat[i]
                scores = results[diversity_scores[i]][results["dataset_name"] == ds].values

                # calculate the correlation coefficient (returns an object)
                result = pearsonr(scores, generalisation_gap)

                ax.scatter(scores, generalisation_gap, color=c, label=ds)
                ax.legend()
                ax.set_xlabel(plot_titles[i] + " {0:.2f}".format(result.statistic))
                #ax.get_xaxis().set_visible(False)
        plt.tight_layout()
        plt.show()

    def plotAccuracy(self, dataset="Test"):
        assert dataset in ["Test", "Validation"], "Please set dataset to either Test or Validation"
        results = pd.read_csv(self.csv_path)

        diversity_scores = ["vs_embed_full_train", "vs_entropy_train", "vs_pixel_train", "vs_inception_train",
                            "n_samples"]
        plot_titles = ["Vendi Score (AE Embedding)", "Label Entropy", "Vendi Score (Raw Pixel)",
                       "Vendi Score (Inception Embedding)", "Number of samples"]

        fig, axes = plt.subplots(nrows=3, ncols=2, sharey=True)

        for c, ds in zip([lred, lblu], ["MNIST", "EMNIST"]):
            if dataset == "Test":
                accuracy = results["test_accuracy"][results["dataset_name"] == ds].values
            elif dataset == "Validation":
                accuracy = results["valid_accuracy"][results["dataset_name"] == ds].values

            for i in range(5):
                ax = axes.flat[i]
                scores = results[diversity_scores[i]][results["dataset_name"] == ds].values

                # calculate the correlation coefficient (returns an object)
                result = pearsonr(scores, accuracy)

                ax.scatter(scores, accuracy, color=c, label=ds + " {0:.2f}".format(result.statistic))
                ax.legend()
                ax.set_xlabel(plot_titles[i])
                # ax.get_xaxis().set_visible(False)
        plt.tight_layout()
        plt.show()


def main():
    #plot = ResultsPlotter(csv_path="results/MNIST_Pixel_VS.csv", metric="vs_pixel")
    #plot.plot()

    #plot = ResultsPlotter(csv_path="results/MNIST_Embed_VS.csv", metric="vs_encoded")
    #plot.plot()

    #plot = ResultsPlotter(csv_path="results/MNIST_Embed_VS_Full.csv", metric="vs_encoded")
    #plot.plot()

    #plot = ResultsPlotter(experiment_name="MNIST_Embed_Inception", metric="vs_encoded")
    #plot.plot()

    plotter = GeneralisationPlotter(experiment_name="GeneralisationMinMaxDiversity")
    plotter.plotAccuracy(dataset="Test")
    plotter.plotAccuracy(dataset="Validation")


if __name__ == "__main__":
    main()