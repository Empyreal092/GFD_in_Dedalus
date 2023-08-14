#!/bin/bash

#SBATCH --nodes=4
#SBATCH --tasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --job-name=2D_Euler
#SBATCH --mail-type=END
#SBATCH --mail-user=sd3201@nyu.edu
#SBATCH --output=slurm_%j.out

module purge

cd /scratch/sd3201/Dedalus_trial/2D_Euler/code
srun /scratch/work/sd3201/dedalus3/run-dedalus-ryanver.bash python 2D_Euler.py 