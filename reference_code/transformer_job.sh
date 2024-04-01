#!/bin/bash
#SBATCH --job-name=moe_en_fr       # name
#SBATCH --nodes=1                    # nodes
#SBATCH --ntasks-per-node=1          # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=64           # number of cores per tasks
#SBATCH --gres=gpu:4                 # number of gpus
#SBATCH --time 48:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=delta_logs/moe-transformer/%x-%j.out           # output file name
#SBATCH --partition=gpuA100x4
#SBATCH --mem=208G
#SBATCH --constraint="scratch"
#SBATCH --gpu-bind=closest   # select a cpu close to gpu on pci bus topology
#SBATCH --account=bbry-delta-gpu
#SBATCH --no-requeue

source virtual_env1/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Just to test

export OMP_NUM_THREADS=60  # if code is not multithreaded, otherwise set to 8 or 16
export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
echo $OMP_NUM_THREADS
echo $GPUS_PER_NODE
echo $MASTER_ADDR
echo $MASTER_PORT

echo "starting training"
echo "waited: $(squeue --Format=PendingTime -j $SLURM_JOB_ID --noheader | tr -d ' ')s"
# echo 'python3 -m torch.distributed.run \
#  --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
#  --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
# transformer/one_layer_transformer_train.py'

echo "slurm_id"
echo "$SLURM_JOBID"
time {
    srun --jobid $SLURM_JOBID bash -c 'python3 -m torch.distributed.run \
    --nproc_per_node $GPUS_PER_NODE --nnodes $SLURM_NNODES --node_rank $SLURM_PROCID \
    --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
    transformer/train.py --data=en_fr'
}
echo "done"
