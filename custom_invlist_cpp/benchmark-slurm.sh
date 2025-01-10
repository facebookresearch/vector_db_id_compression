#!/bin/bash
#SBATCH --job-name=faiss_ivf_benchmark
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH -C bldg1  # Schedule on E5-2698 2.20GHz nodes
#SBATCH -n 1      # Number of tasks (MPI processes)
#SBATCH -c 10     # Number of CPUs per task
#SBATCH -t 48:00:00  # Time limit (HH:MM:SS)
# Activate conda environment
source activate faiss_ivf_compression-graphs
python bench_invlists.py
