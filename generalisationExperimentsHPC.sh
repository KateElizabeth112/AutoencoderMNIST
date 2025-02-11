#!/bin/bash
#PBS -l walltime=12:00:00
#PBS -l select=1:ncpus=15:mem=80gb:ngpus=1:gpu_type=RTX6000
#PBS -N minMaxDiversity_MedNIST_200

# bash script to run generalisation experiments on HPC

cd ${PBS_O_WORKDIR}

# Launch virtual environment
module load anaconda3/personal

# install requirements
#pip install -r requirements.txt

# generate params files
#python generateParamsFiles.py -e "Generalisation_Fixed_Entropy" -r "/rds/general/user/kc2322/home"
python generateParamsFiles.py -e "GeneralisationMinMaxDiversity" -r "/Users/katecevora/Documents/PhD"

# run experiments
#python runExperiment.py  -e "Generalisation_Fixed_Entropy" -r "/rds/general/user/kc2322/home" -s "testGeneralisation.py" -n 500
python runExperiment.py  -e "GeneralisationMinMaxDiversity" -r "/Users/katecevora/Documents/PhD" -s "testGeneralisation.py" -n 200 -d "PneuNIST"
