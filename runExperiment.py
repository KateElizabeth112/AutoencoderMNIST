import subprocess
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

params_folder = "params"
experiment_name = "MNIST_Embed_Inception"
script_name = "VSEmbedInception.py"


def main():
    n_samples_list = [20, 100, 500, 1000, 5000]
    data_category_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, "all"]
    seed = 112

    for n_samples in n_samples_list:
        for data_category in data_category_list:
            params_name = "params_{}_{}_{}.pkl".format(data_category, n_samples, seed)

            command = ["python", script_name, "-e", experiment_name, "-p", params_name]

            # Run the command
            result = subprocess.run(command, capture_output=True, text=True)

            # Print the results
            print("Return code:", result.returncode)
            print("Output:", result.stdout)
            print("Error:", result.stderr)


if __name__ == "__main__":
    main()