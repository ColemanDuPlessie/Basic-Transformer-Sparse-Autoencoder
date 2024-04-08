#!/bin/bash
#SBATCH --job-name=gen_activations_train       # name
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 16:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=delta_logs/gen_activations_train.out           # output file name
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --constraint="scratch"
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bbry-delta-gpu
#SBATCH --no-requeue

module reset # drop modules and explicitly load the ones needed
             # (good job metadata and reproducibility)
             # $WORK and $SCRATCH are now set
module list  # job documentation and metadata

echo "job is starting on `hostname`"

module load python/3.11.6

#source /u/agritsevskiy1/transformer-env/bin/activate

export TORCH_SHOW_CPP_STACKTRACE=1

export OMP_NUM_THREADS=1  # if code is not multithreaded, otherwise set to 8 or 16
# srun -N 1 -n 4 ./a.out > myjob.out
# py-torch example, --ntasks-per-node=1 --cpus-per-task=64
source /u/cduplessie/huggingface-env/bin/activate

export PYTHONPATH=$PYTHONPATH:/u/cduplessie/Basic-Transformer-Sparse-Autoencoder/

pip3 list

CUDA_VISIBLE_DEVICES=0,1,2,3 srun python3 generate_activations.py
