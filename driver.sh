#!/usr/bin/env zsh
#SBATCH --job-name=driver
#SBATCH --partition=instruction
#SBATCH --time=00-00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --output=driver.out
#SBATCH --error=driver.err

module load nvidia/cuda/11.8.0
nvcc driver.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o t
./t
