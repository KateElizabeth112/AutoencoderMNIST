import subprocess
import argparse
import os

# Set up the argument parser
parser = argparse.ArgumentParser(description="Calculate the Vendi score for a dataset from embeddings of an AE model")
parser.add_argument("-e", "--experiment", type=str, help="Name of the experiment.", default="GeneralisationMinMaxDiversity")
parser.add_argument("-r", "--root_dir", type=str, help="Root directory where the code and data are located", default="/Users/katecevora/Documents/PhD")
parser.add_argument("-s", "--script_name", type=str, help="Name of the script to run the expermiment.", default="testGeneralisation.py")
parser.add_argument("-n", "--num_samples", type=int, help="Number of dataset samples per category to use", default=200)
parser.add_argument("-d", "--dataset", type=str, help="Dataset which we will use to run experiments", default="PneuNIST")

args = parser.parse_args()

code_dir = os.path.join(args.root_dir, "code/AutoencoderMNIST")
data_dir = os.path.join(args.root_dir, "data")
params_folder = os.path.join(code_dir, "params")
experiment_name = args.experiment
script_name = args.script_name
dataset = args.dataset


def main():
    n_samples_list = [args.num_samples]
    seeds = [112, 234, 23, 453, 21, 12, 6, 2, 67, 88]

    for n_samples in n_samples_list:
        for s in seeds:
            data_category = "all"

            if experiment_name == "Generalisation_Fixed_Entropy":
                params_name = "params_{}_{}_{}_{}.pkl".format(data_category, n_samples, s, dataset)

                print(
                    "Running experiment with configuration: category={0}, n_samples={1}, seed={2}, dataset={3}".format(
                        data_category, n_samples, s, dataset))

                command = ["python", script_name, "-e", experiment_name, "-p", params_name, "-r", args.root_dir]

                # Run the command
                result = subprocess.run(command, capture_output=True, text=True)

            elif experiment_name == "GeneralisationMinMaxDiversity":
                for diversity in ["high", "low"]:
                    params_name = "params_{}_{}_{}_{}_{}.pkl".format(data_category, n_samples, s, dataset, diversity)


                    print(
                        "Running {4} experiment with configuration: category={0}, n_samples={1}, seed={2}, dataset={3}".format(
                            data_category, n_samples, s, dataset, experiment_name))

                    command = ["python", script_name, "-e", experiment_name, "-p", params_name, "-r", args.root_dir]

                    # Run the command
                    result = subprocess.run(command, capture_output=True, text=True)
                    print(result.stdout)
                    print(result.stderr)



if __name__ == "__main__":
    main()
