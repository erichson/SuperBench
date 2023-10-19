#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=dasrepo_g
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=125
#SBATCH --module=gpu,nccl-2.15
#SBATCH --mail-user=joey000122@gmail.com
#SBATCH --mail-type=ALL



module load pytorch/2.0.1

env=/global/homes/j/juny012/.local/perlmutter/pytorch2.0.1

cmd1="srun --exact -n 1 -G1 -c125 --gpu-bind=map_gpu:0 python train_FNO.py --data_path /pscratch/sd/j/junyi012/superbench_v1/nskt16000_1024 --data_name nskt_16k --epochs 500 --batch_size 64"


set -x
    bash -c "
    $cmd1
    "