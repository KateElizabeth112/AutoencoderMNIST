# A script to generate and save training params files as pickles
import os
import pickle as pkl

params_folder = "params"
experiment_name = "MNIST_Pixel_VS"


def main():
    # check whether we already have a params folder, if not make one
    if not os.path.exists(params_folder):
        os.mkdir(params_folder)

    # check whether we have a folder for the experiment
    if not os.path.exists(os.path.join(params_folder, experiment_name)):
        os.mkdir(os.path.join(params_folder, experiment_name))

    # save our first params file as a test
    params = {
        "data_category": 5,
        "n_samples": 500,
        "random_seed": 112,
        "n_layers": 3,
        "n_epochs": 20,
        "n_workers": 0,
        "batch_size": 20,
        "model_name": "autoencoderMNIST.pt"
    }

    params_name = "test_params.pkl"

    f = open(os.path.join(params_folder, experiment_name, params_name), "wb")
    pkl.dump(params, f)
    f.close()


if __name__ == "__main__":
    main()
