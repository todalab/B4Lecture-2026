#!/bin/bash
#SBATCH --job-name=forB4Lecture    # Job name
#SBATCH --time=24:00:00             # Time limit hrs:min:sec
#SBATCH --output=logs/output/job_output_%j.log
#SBATCH --error=logs/error/job_error_%j.log
#SBATCH --gres=gpu:1                # Number of GPUs to use

# Go to the project directory
PROJECT_DIR=/nas01/homes/hagiwara26-1000093/B4Lecture-2026/ex8/f_hagiwara

cd "$PROJECT_DIR" || exit 1

source .venv/bin/activate

python3 main.py datadir="$PROJECT_DIR/data"

echo "Training job completed."