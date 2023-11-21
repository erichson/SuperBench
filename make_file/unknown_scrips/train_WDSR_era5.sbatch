#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=dasrepo_g
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --module=gpu,nccl-2.15
#SBATCH --mail-user=joey000122@gmail.com
#SBATCH --mail-type=ALL



module load pytorch/2.0.1

env=/global/homes/j/juny012/.local/perlmutter/pytorch2.0.1

cmd1="srun --exact -n 1 -G1 --gpu-bind=map_gpu:0 python train.py --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --data_name era5 --epochs 200 --model WDSR --in_channels 3 --upscale_factor 16 --batch_size 256 --lr 0.0005"
cmd2="srun --exact -n 1 -G1 --gpu-bind=map_gpu:1 python train.py --data_path /pscratch/sd/j/junyi012/superbench_v2/era5 --data_name era5 --epochs 200 --model WDSR --upscale_factor 8 --in_channels 3 --batch_size 256 --lr 0.0005" 

set -x
    bash -c "
    $cmd1 & $cmd2 &
    wait
    "