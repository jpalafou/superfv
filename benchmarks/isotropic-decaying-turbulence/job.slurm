#!/bin/bash
#SBATCH --job-name=isoturb                  # create a short name for your job
#SBATCH --nodes=1                           # node count
#SBATCH --ntasks-per-node=1                 # total number of tasks per node
#SBATCH --cpus-per-task=1                   # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G                    # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00                     # total run time limit (HH:MM:SS)
#SBATCH --mail-user=jp7427@princeton.edu    # email address for notifications
#SBATCH --mail-type=ALL                     # notifications for job done & fail
#SBATCH --output=out/slurm-%j.out           # output file

module purge
module load anaconda3/2025.6
conda activate superfv

python benchmarks/isotropic-decaying-turbulence --N 128
