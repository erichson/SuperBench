#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=dasrepo_g
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --module=gpu,nccl-2.15
#SBATCH --mail-user=joey000122@gmail.com
#SBATCH --mail-type=ALL



module load pytorch/2.0.1

env=/global/homes/j/juny012/.local/perlmutter/pytorch2.0.1

cmd1="srun --exact -n 1 -G1 -c 32 --gpu-bind=map_gpu:0 python train.py --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --data_name nskt_16k --epochs 300 --model SwinIR --in_channels 3 --upscale_factor 8 --batch_size 64 --lr 0.0001" 
cmd1="srun --exact -n 1 -G1 -c 32 --gpu-bind=map_gpu:1 python train.py --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k --data_name nskt_16k --epochs 300 --model SwinIR --in_channels 3 --upscale_factor 16 --batch_size 64 --lr 0.0001" 
cmd1="srun --exact -n 1 -G1 -c 32 --gpu-bind=map_gpu:2 python train.py --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --data_name nskt_32k --epochs 300 --model SwinIR --in_channels 3 --upscale_factor 8 --batch_size 64 --lr 0.0001" 
cmd1="srun --exact -n 1 -G1 -c32 --gpu-bind=map_gpu:3 python train.py --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_32k --data_name nskt_32k --epochs 300 --model SwinIR --in_channels 3 --upscale_factor 16 --batch_size 64 --lr 0.0001" 
set -x
    bash -c "
    $cmd1 & $cmd2 & $cmd3 & $cmd4 &
    wait
    "