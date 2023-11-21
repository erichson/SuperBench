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

cmd1="srun --exact -n 1 -G1 --gpu-bind=map_gpu:0 python train.py --data_path /pscratch/sd/j/junyi012/superbench_v1/cosmo_2048 --data_name cosmo --epochs 300 --model SRCNN --in_channels 2 --out_channels 2 --upscale_factor 8 --batch_size 128 --lr 0.001"
cmd2="srun --exact -n 1 -G1 --gpu-bind=map_gpu:1 python train.py --data_path /pscratch/sd/j/junyi012/superbench_v1/nskt_16000_1024 --data_name nskt_16k --epochs 300 --model SRCNN --upscale_factor 8 --in_channels 3 --batch_size 128 --lr 0.001" 
cmd3="srun --exact -n 1 -G1 --gpu-bind=map_gpu:2 python train.py --data_path /pscratch/sd/j/junyi012/superbench_v1/nskt_32000_1024 --data_name nskt_32k --epochs 300 --model SRCNN --upscale_factor 8 --in_channels 3 --batch_size 128 --lr 0.001" 
cmd4="srun --exact -n 1 -G1 --gpu-bind=map_gpu:3 python train.py --data_path /pscratch/sd/j/junyi012/superbench_v1/era5 --data_name era5 --epochs 300 --model SRCNN --upscale_factor 8 --in_channels 3 --batch_size 128 --lr 0.001" 
set -x
    bash -c "
    $cmd1 & $cmd2 & $cmd3 & $cmd4 &
    wait
    "