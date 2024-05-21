# tools to plot the results
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, csv_path=""):
        self.csv_path = csv_path

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


def main():
    plot = Plotter(csv_path="results/MNIST_Pixel_VS.csv")
    plot.plot()


if __name__ == "__main__":
    main()