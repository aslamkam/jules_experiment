#!/bin/bash
#SBATCH --job-name=random_forest_run       # Job name
#SBATCH --account=def-wsmith-ac #ctb-wsmith #def-wsmith-ac #def-wsmith-ac ##SBATCH --gres=gpu:p100:1
#comment-SBATCH --partition=gpu
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=1000
#SBATCH --output=output.log     # Standard output file
#SBATCH --error=error.log       # Error output file

# Load required modules
# # module load python/3.10
# # module load scipy-stack                   # SciPy stack for NumPy, pandas, etc.

# Option 1: Use a virtual environment with `seaborn` installed
source /home/kaslam/scratch/tf_venv/bin/activate               # Replace `myenv` with your environment name

# Navigate to your project directory
cd /home/kaslam/scratch/IUPAC/GSP-main/GSP-main/Main/Python/IuPac_Iteration_v2.py

# Run your application
# python RandomForests_HyperParameter_Tuning_RandomizedGridSearch_V2.py
python IuPac_Iteration_v2.py
