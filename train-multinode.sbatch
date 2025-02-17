#!/bin/sh

# Train with 2 nodes, 8 cards each
#SBATCH --job-name=train-glm        # name
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node 8
#SBATCH --time 96:00:00              # maximum execution time (HH:MM:SS)
#SBATCH --output=%x-%j.out           # output file name
#SBATCH --account ENT214007
#SBATCH --partition=gp4d

export GPUS_PER_NODE=8
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=9901
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

srun --ntasks=2 --ntasks-per-node=1 bash -c 'singularity run --bind /work/u3937558/:/work/u3937558/ --nv /home/u3937558/deepspeed bash singularity-run.sh $GPUS_PER_NODE $SLURM_NNODES $SLURM_PROCID $MASTER_ADDR $MASTER_PORT'
