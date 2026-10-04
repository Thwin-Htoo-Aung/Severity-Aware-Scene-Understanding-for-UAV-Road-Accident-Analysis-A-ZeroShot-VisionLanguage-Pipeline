#!/bin/bash

#SBATCH -A IscrC_ISAAC
#SBATCH -p boost_usr_prod
#SBATCH --qos normal
#SBATCH --time=01:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=123000
#SBATCH --job-name=DroneAmbulance
#SBATCH --out=out.log
#SBATCH --err=out.log

export HF_HOME=$SCRATCH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1  

srun python qwen_yolo.py $@