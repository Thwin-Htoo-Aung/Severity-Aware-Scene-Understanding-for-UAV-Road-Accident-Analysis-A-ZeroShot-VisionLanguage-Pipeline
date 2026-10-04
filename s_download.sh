#!/bin/bash

#SBATCH -A IscrC_ISAAC
#SBATCH -p lrd_all_serial
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=DroneAmbulance
#SBATCH --out=download.log
#SBATCH --err=download.log

srun python test.py