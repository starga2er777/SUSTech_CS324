#!/bin/bash
#SBATCH -o job.%j.out          
#SBATCH --partition=a100
#SBATCH --qos=a100
#SBATCH -J myFirstGPUJob
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=6
#SBATCH --gres=gpu:1
#SBATCH --job-name=pytorch

cd /home/wangdx_lab/cse12012524/SUSTech_CS324/assignment3/Part\ 1/
python train.py