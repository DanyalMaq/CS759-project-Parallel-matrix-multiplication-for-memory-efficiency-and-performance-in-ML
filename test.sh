#!/usr/bin/env zsh
#SBATCH --job-name=test
#SBATCH --partition=instruction
#SBATCH --time=00-00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:2
#SBATCH --mem=20G
#SBATCH --output=test.out
#SBATCH --error=test.err

module load nvidia/cuda/11.8.0
nvcc test_time.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -lcurand -lcublas -o managed
nvcc test_manual.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -lcurand -lcublas -o manual
nvcc test_async.cu matmul.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -lcurand -lcublas -o async


for ((power = 5; power <= 12; power++)); do
    argument=$((2**power))
    ./managed $argument $argument $argument 2
    ./manual $argument $argument $argument 2
    ./async $argument 2
done
