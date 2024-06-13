# tools to plot the results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


class ResultsPlotter:
    """
    Class of methods for plotting results - the vendi score calculated for different dataset sizes.
    :param metric: String specifying the metric to plot.
    :param csv_path: String path specifying then results CSV.
    """
    def __init__(self, metric="", csv_path=""):
        # Check the types of the arguments are as expected
        assert os.path.exists(csv_path), "Path {} to results CSV does not exist.".format(csv_path)
        assert isinstance(metric, str), "Please specify the metric to be plotted, {}, as a string.".format(metric)

        self.csv_path = csv_path
        self.metric = metric

        # Run some checks on the value of metric
        if metric == "vs_pixel":
            self.plot_title = "MNIST Pixel Vendi Score"
        elif metric == "vs_encoded":
            self.plot_title = "MNIST Embed Vendi Score"
        elif metric == "":
            raise ValueError("Please set the metric to be plotted.")
        else:
            raise ValueError("The metric for plotting is not recognised.")

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
        ax.set_title(self.plot_title)
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


def main():
    plot = ResultsPlotter(csv_path="results/MNIST_Pixel_VS.csv", metric="vs_pixel")
    plot.plot()

    #plot = ResultsPlotter(csv_path="results/MNIST_Embed_VS.csv", metric="vs_encoded")
    #plot.plot()

    plot = ResultsPlotter(csv_path="results/MNIST_Embed_VS_Full.csv", metric="vs_encoded")
    plot.plot()


if __name__ == "__main__":
    main()