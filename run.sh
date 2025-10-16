#!/bin/bash
#SBATCH --job-name=myjob_array
#SBATCH --output=experiments/results/output_%A_%a.txt
#SBATCH --error=experiments/results/error_%A_%a.txt
#SBATCH -c 10
#SBATCH -t 20:00:00
#SBATCH -p gpu_quad
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=15G

# Set project directory
PROJECT_DIR="/n/data2/hms/dbmi/beamlab/manqing/mini-project"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"

# Use HF_HOME only (not TRANSFORMERS_CACHE - deprecated)
export HF_HOME="${PROJECT_DIR}/hf_cache"
export HF_DATASETS_CACHE="${PROJECT_DIR}/hf_cache/datasets"
export TORCH_HOME="${PROJECT_DIR}/torch_cache"

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TORCH_HOME

export PYTHONPATH="${PYTHONPATH}:/n/data2/hms/dbmi/beamlab/manqing/mini-project"
# Load modules FIRST - UNCOMMENT these lines and move them to the top
module load gcc/14.2.0
module load python/3.13.1
#module load cuda/12.8

#rm -rf myenv
#
## Create and activate the virtual environment
#python3 -m venv myenv
source myenv/bin/activate


# Install requirements inside the virtual environment
#pip3 install -r requirements.txt


python src/generate_data.py