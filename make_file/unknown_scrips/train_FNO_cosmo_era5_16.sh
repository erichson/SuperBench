#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=dasrepo_g
#SBATCH -C gpu
#SBATCH --nodes=1
#SBATCH -q regular
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --module=gpu,nccl-2.15
#SBATCH --mail-user=joey000122@gmail.com
#SBATCH --mail-type=ALL



module load pytorch/2.0.1

env=/global/homes/j/juny012/.local/perlmutter/pytorch2.0.1

cmd1="srun --exact -n 1 -G1 -c64 --gpu-bind=map_gpu:0 python train_FNO.py --data_path /pscratch/sd/j/junyi012/superbench_v1/cosmo_2048 --data_name cosmo --epochs 500 --modes 12 --in_channels 2 --out_channels 2 --upscale_factor 16"
cmd2="srun --exact -n 1 -G1 -c64 --gpu-bind=map_gpu:1 python train_FNO.py --data_path /pscratch/sd/j/junyi012/superbench_v1/era5 --data_name era5 --epochs 500 --modes 12 --in_channels 3 --out_channels 3 --upscale_factor 16"

set -x
    bash -c "
    $cmd1 & $cmd2 &
    wait
    "