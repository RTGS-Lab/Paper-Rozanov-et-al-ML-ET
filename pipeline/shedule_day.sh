#!/bin/bash -l
#SBATCH --job-name=ET
#SBATCH --account=runck014
#SBATCH --output=./logs/out_%A_%a.txt
#SBATCH --error=./logs/err_%A_%a.txt
#SBATCH --time=01:30:00
#SBATCH --partition=msismall#msigpu
#SBATCH --mem=100G
#SBATCH --cpus-per-task=2
#SBATCH --array=1-12 #1-2211%50          # Skip first 30, process files 31-2211

source ~/.bashrc
conda activate ET_lightgbm

#files=($(find /home/runck014/shared/et_upscaling/MODIS/2024 -name "MODIS_2024_*.tif" | sort))
mapfile -t files < remaining_files.txt

# Get the file for this array task (SLURM_ARRAY_TASK_ID is already 31-2211)
filepath=${files[$SLURM_ARRAY_TASK_ID-1]}

# Extract just the filename (no path)
fname=$(basename "$filepath")

python3 pipeline.py --fname "$fname"
