#!/bin/bash
#SBATCH --job-name=profile       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=4G                 # total memory per node
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module purge
module load cudatoolkit/12.9
module load anaconda3/2024.10
conda activate superfv

ncu -o my_report_${SLURM_JOBID} python benchmarks/advection_profiling
