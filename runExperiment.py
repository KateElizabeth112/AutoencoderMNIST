import subprocess
import warnings

# Suppress all warnings
warnings.filterwarnings('ignore')

params_folder = "params"
experiment_name = "Generalisation_Fixed_Entropy"
script_name = "testGeneralisation.py"


def main():
    n_samples_list = [20, 100, 500, 1000, 5000]
    seeds = [112, 234, 23, 453, 21, 12, 6, 2, 67, 88]

    for n_samples in n_samples_list:
        for s in seeds:
            for dataset in ["MNIST", "EMNIST"]:
                data_category = "all"
                params_name = "params_{}_{}_{}_{}.pkl".format(data_category, n_samples, s, dataset)

                print(
                    "Running experiment with configuration: category={0}, n_samples={1}, seed={2}, dataset={3}".format(
                        data_category, n_samples, s, dataset))

                command = ["python", script_name, "-e", experiment_name, "-p", params_name]

                # Run the command
                result = subprocess.run(command, capture_output=True, text=True)

                # Print the results
                print("Return code:", result.returncode)
                print("Output:", result.stdout)
                print("Error:", result.stderr)


if __name__ == "__main__":
    main()
