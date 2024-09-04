#!/bin/bash

# Job Name
#SBATCH --job-name=pytorch-gpu
# Number of Nodes
#SBATCH --nodes=1
# Number of processes per Node
#SBATCH --ntasks-per-node=1
# Number of CPU-cores per task
#SBATCH --cpus-per-task=4
# Set the GPU-Partition (opt. but recommended)
#SBATCH --partition=gpu
# Allocate node with certain GPU
##SBATCH --gres=gpu:a100:3  #gruenau10 is not working, wrong nameing?
# Allocate any available GPU
#SBATCH --gres=gpu:2

# Request 32 GB of total memory for the job
#SBATCH --mem=32G

#module load cuda

#python3 finetuning.py --vuln_type=xsrf  --cache_data=../cache_data/xsrf --save_dir=../saved_models/xsrf --per_device_train_batch_size=1 --per_device_eval_batch_size=1
srun python3 finetuning.py --vuln_type=xsrf  --cache_data=../cache_data/xsrf --save_dir=../saved_models/xsrf --per_device_train_batch_size=1 --per_device_eval_batch_size=1