# tools to plot the results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class Plotter:
    def __init__(self, csv_path=""):
        self.csv_path = csv_path

    def plotPixelVS(self):
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
                results_cat = results[(results["data_category"] == cat) & (results["n_samples"] == N)]["vs_pixel"].values
                values.append(results_cat[0])

            # plot the series for a category
            ax.plot(n_samples, values, label="{}".format(cat))

        ax.legend()
        ax.set_title("MNIST Pixel Vendi Score")
        ax.set_xticks(n_samples, n_samples)
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Vendi Score")
        ax.set_xscale('log')
        plt.show()

    def plotEmbedVS(self):
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
                results_cat = results[(results["data_category"] == cat) & (results["n_samples"] == N)][
                    "vs_encoded"].values
                values.append(results_cat[0])

            # plot the series for a category
            ax.plot(n_samples, values, label="{}".format(cat))

        ax.legend()
        ax.set_title("MNIST Embed Vendi Score")
        ax.set_xticks(n_samples, n_samples)
        ax.set_xlabel("Dataset Size")
        ax.set_ylabel("Vendi Score")
        ax.set_xscale('log')
        plt.show()


class LossPlotter:
    """
    Class of methods for plotting loss curves.

    :param epochs: List of epochs where the loss was recorded
    :param loss: List of loss values averaged over training/test set
    :param save_path: Path where the loss plot will be saved.
    """
    def __init__(self, epochs, loss, save_path=""):
        self.epochs = epochs
        self.loss = loss
        self.save_path = save_path

        # run some checks
        assert isinstance(epochs, list), "The epochs variable must be of type list"
        assert isinstance(loss, list), "The loss variable must be of type list"
        assert isinstance(save_path, str), "The save_path variable must be of type str"
        assert os.path.exists(save_path), "The save_path {} does not exist".format(save_path)

    """
    Plot the training loss curve.
    """
    def plotTrainLoss(self):
        plt.plot(self.epochs, self.loss)
        plt.xlabel("Epochs")
        plt.ylabel("Train Loss")

        if self.save_path == "":
            plt.show()
        else:
            plt.savefig(os.path.join(self.save_path, "loss.png"))


def main():
    plot = Plotter(csv_path="results/MNIST_Pixel_VS.csv")
    plot.plotPixelVS()

    plot = Plotter(csv_path="results/MNIST_Embed_VS.csv")
    plot.plotEmbedVS()

    plot = Plotter(csv_path="results/MNIST_Embed_VS_Partial.csv")
    plot.plotEmbedVS()

if __name__ == "__main__":
    main()