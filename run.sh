#!/bin/bash
#SBATCH --job-name=myjob_array
#SBATCH --output=experiments/results/output_%A_%a.txt
#SBATCH --error=experiments/results/error_%A_%a.txt
#SBATCH -c 10
#SBATCH -t 6:00:00
#SBATCH -p mit_normal_gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=15G

cd $SLURM_SUBMIT_DIR
# Get the directory where this script is located
SCRIPT_DIR=$SLURM_SUBMIT_DIR

# Load configuration from config.json
CONFIG_FILE="${SCRIPT_DIR}/config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: config.json not found at $CONFIG_FILE"
    echo "Please copy config.example.json to config.json and update with your settings"
    exit 1
fi

# Parse JSON config using python
PROJECT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['project_dir'])")
HF_HOME_TEMPLATE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['cache_dirs']['hf_home'])")
HF_DATASETS_TEMPLATE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['cache_dirs']['hf_datasets'])")
TORCH_HOME_TEMPLATE=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['cache_dirs']['torch_home'])")

# Expand ${PROJECT_DIR} in template paths
export HF_HOME="${HF_HOME_TEMPLATE//\$\{PROJECT_DIR\}/$PROJECT_DIR}"
export HF_DATASETS_CACHE="${HF_DATASETS_TEMPLATE//\$\{PROJECT_DIR\}/$PROJECT_DIR}"
export TORCH_HOME="${TORCH_HOME_TEMPLATE//\$\{PROJECT_DIR\}/$PROJECT_DIR}"

# Set Python path
export PYTHONPATH="${PYTHONPATH}:${PROJECT_DIR}"

# Create cache directories
mkdir -p $HF_HOME
mkdir -p $HF_DATASETS_CACHE
mkdir -p $TORCH_HOME

echo "Configuration loaded:"
echo "  PROJECT_DIR: $PROJECT_DIR"
echo "  HF_HOME: $HF_HOME"
echo "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
echo "  TORCH_HOME: $TORCH_HOME"
# Load modules FIRST - UNCOMMENT these lines and move them to the top
# module load gcc/14.2.0
#module load cuda/12.8
module load miniforge/24.3.0-0

# rm -rf myenv
#
## Create and activate the virtual environment
# python3 -m venv myenv
source myenv/bin/activate


# Install requirements inside the virtual environment
# pip3 install -r requirements.txt


python scripts/generate_data.py
