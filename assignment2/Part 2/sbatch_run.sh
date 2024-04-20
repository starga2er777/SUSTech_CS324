#!/bin/bash
#SBATCH -o job.%j.out          
#SBATCH --partition=gpulab02
#SBATCH --qos=gpulab02
#SBATCH -J myFirstGPUJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=pytorch

nvidia-smi

cd /home/wangdx_lab/cse12012524/SUSTech_CS324/assignment2/Part\ 2/
python train_script.py