#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=dasrepo_g
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH -q shared
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --module=gpu,nccl-2.15
#SBATCH --mail-user=joey000122@gmail.com
#SBATCH --mail-type=ALL



module load pytorch/2.0.1

env=/global/homes/j/juny012/.local/perlmutter/pytorch2.0.1

cmd1="srun --exact -n 1 -G1 -c32 ython train_FNO.py --data_path /pscratch/sd/j/junyi012/superbench_v1/nskt16000_1024 --data_name nskt_16k --epochs 500 --modes 12 --crop_size 1024 --n_patches 1 --batch_size 8"

cmd2="srun --exact -n 1 -G1 -c32 python train_FNO.py --data_path /pscratch/sd/j/junyi012/superbench_v1/nskt32000_1024 --data_name nskt_32k --epochs 500 --modes 12  --n_patches 1 --batch_size 8"
set -x
    bash -c "
    $cmd1 & $cmd2 
    wait
    "