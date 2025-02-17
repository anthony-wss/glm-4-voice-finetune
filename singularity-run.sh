GPUS_PER_NODE=$1
SLURM_NNODES=$2
SLURM_PROCID=$3
MASTER_ADDR=$4
MASTER_PORT=$5

source /home/u3937558/.bashrc
conda activate deepspeed

echo "torchrun --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py --deepspeed ds_config.json"
torchrun --nproc_per_node $GPUS_PER_NODE --nnodes=$SLURM_NNODES --node_rank=$SLURM_PROCID --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py --deepspeed ds_config.json
