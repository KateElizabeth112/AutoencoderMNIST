# tools to plot the results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset
from scipy.stats import pearsonr, ConstantInputWarning
import warnings

warnings.simplefilter("ignore", ConstantInputWarning)

lblu = "#add9f4"
lred = "#f36860"
lgrn = "#7dda7e"

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


class ResultsProcesser:
    """
    Class for plotting results of generalisation experiments.
    """
    def __init__(self, experiment_name=""):
        # Check the types of the arguments are as expected
        assert isinstance(experiment_name, str), "The experiment name must be a string"
        assert os.path.exists(os.path.join("./results", experiment_name + ".csv")), "Path to results CSV does not exist."

        self.experiment_name = experiment_name
        self.csv_path = os.path.join("./results", experiment_name + ".csv")
        self.results = pd.read_csv(self.csv_path)

        # Run some checks on the value of experimennt
        if self.experiment_name not in ("Generalisation_Fixed_Entropy", "GeneralisationMinMaxDiversity"):
            raise ValueError("The experiment {} does not exist".format(self.experiment_name))

        # create a list of diversity score metrics
        score_titles = ["VS", "Av. Sim.", "IntDiv"]
        scores = ["vs", "av_sim", "intdiv"]
        embed_titles = [" (Raw Pixel)", " (AE)", " (Inception)"]
        embed = ["pixel", "embed_full", "inception"]

        plot_titles = []
        diversity_scores = []

        for k in range(3):
            for j in range(3):
                plot_titles.append(score_titles[j] + embed_titles[k])
                diversity_scores.append("{0}_{1}_train".format(scores[j], embed[k]))

        # Add number of samples and label entropy
        plot_titles.append("Number of Samples")
        plot_titles.append("Label Entropy")
        diversity_scores.append("n_samples")
        diversity_scores.append("vs_entropy_train")

        self.diversity_scores = diversity_scores
        self.plot_titles = plot_titles


    def plot(self, metric="test_accuracy", dataset=""):
        assert metric in ["test_accuracy", "valid_accuracy", "generalisation_gap"], \
            "Please set the plotting metric to either 'test_accuracy' or 'valid_accuracy' or 'generalisation_gap'"

        assert isinstance(dataset, list), "Please specify the dataset/s within a list"

        # Check that the specified dataset/s are present in the results file
        for ds in dataset:
            assert ds in list(np.unique(self.results["dataset_name"].values)), "Dataset {} not found in results".format(ds)

        fig, axes = plt.subplots(nrows=4, ncols=3, sharey=True)

        # list of colours to use for plotting different datasets
        colours_list = [lred, lblu, lgrn]

        # if a dataset was not specified, get a list of datasets from the results
        if dataset == "":
            dataset_names = list(np.unique(self.results["dataset_name"].values))
        else:
            dataset_names = list(dataset)

        num_datasets = len(dataset_names)
        colours = colours_list[:num_datasets]

        for c, ds in zip(colours, dataset_names):
            if metric in ["test_accuracy", "valid_accuracy"]:
                accuracy = self.results[metric][self.results["dataset_name"] == ds].values
            elif metric == "generalisation_gap":
                valid_accuracy = self.results["valid_accuracy"][self.results["dataset_name"] == ds].values
                test_accuracy = self.results["test_accuracy"][self.results["dataset_name"] == ds].values
                accuracy = test_accuracy - valid_accuracy / (0.5 * (test_accuracy + valid_accuracy))
            else:
                print("Metric {} not recognised".format(metric))

            for i in range(len(self.diversity_scores)):
                ax = axes.flat[i]
                print(self.diversity_scores[i])
                diversity = self.results[self.diversity_scores[i]][self.results["dataset_name"] == ds].values

                # Find out if we have any Nan values in scores (due to missing data)
                nan_idx = np.isnan(diversity)

                # filter out nan entries
                diversity_nonan = diversity[np.invert(nan_idx)]
                accuracy_nonnan = accuracy[np.invert(nan_idx)]

                # Check whether we have any data for this metric
                if diversity_nonan.shape[0] > 0:
                    # calculate the correlation coefficient (returns an object)
                    result = pearsonr(diversity_nonan, accuracy_nonnan)
                    ax.scatter(diversity_nonan, accuracy_nonnan, color=c, label=ds + " {0:.2f} ({1:.2f})".format(result.statistic, result.pvalue))
                    ax.legend()
                ax.set_xlabel(self.plot_titles[i])
        plt.tight_layout()
        plt.show()

    def __printCorrelation__(self, diversity, accuracy):
        """
        Helper function for calculating and printing the correlation between diversity scores and test accuracy.
        :param diversity:
        :param accuracy:
        :return:
        """
        # filter out the nans
        nan_idx = np.isnan(diversity)
        diversity_nonan = diversity[np.invert(nan_idx)]
        accuracy_nonnan = accuracy[np.invert(nan_idx)]

        # calculate correlation coefficient if we have any data
        if diversity_nonan.shape[0] > 0:
            # calculate the correlation coefficient (returns an object)
            corr = pearsonr(diversity_nonan, accuracy_nonnan)

            if corr.pvalue < 0.05:
                pval = "*"
            elif corr.pvalue < 0.01:
                pval = "**"
            else:
                pval = ""

            print("& {0:.2f}{1} ".format(abs(corr.statistic), pval), end="")
        else:
            print("& ", end="")

    def printResults(self, output="test_accuracy"):
        """
        Print a table of results in latex format and save to a text file if specified
        :return:
        """
        assert output in ["test_accuracy", "valid_accuracy", "generalisation_gap"], \
            "Please set the plotting metric to either 'test_accuracy' or 'valid_accuracy' or 'generalisation_gap'"

        # Get the names of the datasets present in results. We will have a separate column for each dataset
        dataset_names = np.unique(self.results["dataset_name"].values)

        # print the first few lines of the latex table
        print(r"\begin{tabular}{p{3.2cm}|p{0.6cm}p{0.6cm}p{0.6cm}p{0.6cm}|p{0.6cm}p{0.6cm}p{0.6cm}p{0.6cm}|p{0.6cm}p{0.6cm}p{0.6cm}p{0.6cm}|}")
        print(r" &  \multicolumn{4}{|c|}{MNIST} & \multicolumn{4}{|c|}{EMNIST} &\multicolumn{4}{|c|}{PneuMNIST}\\")
        print(r"\hline")
        print(r"No. Samples & 500 & 1000 & 2000 & all & 500 & 1000 & 2000 & all & 200 & 500 & 1000 & all \\")
        print(r"\hline")

        # iterate over the diversity scoring metrics
        for score, score_name in zip(self.diversity_scores, self.plot_titles):
            print(score_name, end="")
            # iterate over the datasets
            for dataset_name in dataset_names:
                # find the range of dataset sizes used for this dataset
                n_samples = np.unique(self.results["n_samples"][self.results["dataset_name"] == dataset_name].values)

                # iterate over the number of samples
                for ns in n_samples:
                    # filter the results by dataset, diversity metric and number of samples
                    condition_1 = self.results["dataset_name"] == dataset_name
                    condition_2 = self.results["n_samples"] == ns
                    diversity = self.results[score][condition_1 & condition_2]
                    accuracy = self.results[output][condition_1 & condition_2]

                    self.__printCorrelation__(diversity, accuracy)

                # Get a correlation value for all samples
                condition_1 = self.results["dataset_name"] == dataset_name
                diversity = self.results[score][condition_1]
                accuracy = self.results[output][condition_1]

                self.__printCorrelation__(diversity, accuracy)

                print("", end="")
            print("\\\\")

        # print the last part of the table in latex
        print(r"\end{tabular}")

def main():
    plotter = ResultsProcesser(experiment_name="GeneralisationMinMaxDiversity")
    #plotter.plotAccuracy(metric="test_accuracy", dataset=["MNIST", "EMNIST", "PneuNIST"])

    plotter.printResults(output="test_accuracy")


if __name__ == "__main__":
    main()