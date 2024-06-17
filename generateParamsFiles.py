# A script to generate and save training params files as pickles
import os
import pickle as pkl
import random

params_folder = "params"
experiment_name = "MNIST_Embed_Inception"


def main():
    # check whether we already have a params folder, if not make one
    if not os.path.exists(params_folder):
        os.mkdir(params_folder)

    # check whether we have a folder for the experiment
    if not os.path.exists(os.path.join(params_folder, experiment_name)):
        os.mkdir(os.path.join(params_folder, experiment_name))

    n_samples_list = [20, 100, 500, 1000, 5000]
    data_category_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "all"]
    seed = 112

    for n_samples in n_samples_list:
        for data_category in data_category_list:
            params_name = "params_{}_{}_{}.pkl".format(data_category, n_samples, seed)

            # generate a unique ID for the model
            unique_id = ''.join(random.choices('0123456789', k=6))

            if experiment_name == "MNIST_Embed_VS_Partial":
                # save params file
                params = {
                    "data_category": data_category,
                    "n_samples": n_samples,
                    "random_seed": seed,
                    "n_layers": 3,
                    "n_epochs": 150,
                    "n_workers": 0,
                    "batch_size": 20,
                    "model_name": "autoencoderMNISTpartial_{}.pt".format(unique_id)
                }
            elif experiment_name == "MNIST_Embed_VS_Full":
                params = {
                    "data_category": data_category,
                    "n_samples": n_samples,
                    "random_seed": seed,
                    "n_layers": 3,
                    "n_epochs": 100,
                    "n_workers": 0,
                    "batch_size": 20,
                    "model_name": "autoencoderMNISTfull_339108.pt"
                }
            elif experiment_name == "MNIST_Pixel_VS":
                params = {
                    "data_category": data_category,
                    "n_samples": n_samples,
                    "random_seed": seed,
                    "n_layers": 0,
                    "n_epochs": 0,
                    "n_workers": 0,
                    "batch_size": 20,
                    "model_name": None
                }
            elif experiment_name == "MNIST_Embed_Inception":
                params = {
                    "data_category": data_category,
                    "n_samples": n_samples,
                    "random_seed": seed,
                    "n_layers": 0,
                    "n_epochs": 0,
                    "n_workers": 0,
                    "batch_size": 20,
                    "model_name": None
                }
            else:
                print("Experiment name is not recognised")

            f = open(os.path.join(params_folder, experiment_name, params_name), "wb")
            pkl.dump(params, f)
            f.close()


if __name__ == "__main__":
    main()
