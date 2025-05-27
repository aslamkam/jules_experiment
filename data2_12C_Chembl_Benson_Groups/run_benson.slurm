#!/bin/bash
#SBATCH --job-name=install_pandas
#SBATCH --account=def-wsmith-ac
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=install_output.log
#SBATCH --error=install_error.log

module load StdEnv/2023 apptainer/1.3.5

# Set up a host-side virtualenv
VENV_DIR=/home/kaslam/scratch/rmg_venv
python3 -m venv $VENV_DIR
source $VENV_DIR/bin/activate

# Install pandas
pip install pandas tqdm

# Run the script inside the container using the host Python + VENV
apptainer exec \
    --bind /home/kaslam/scratch/myrmgfiles:/rmg/RMG-Py/myrmgfiles \
    rmg-with-modules.sif \
    python /rmg/RMG-Py/myrmgfiles/add_benson_group_to_csv.py
