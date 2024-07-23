#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --account=dasrepo_g
#SBATCH --job-name=nskt_16k_sim_WDSR_4
#SBATCH -C gpu
#SBATCH -q shared
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --module=gpu,nccl-2.15
#SBATCH --mail-user=Joey000122@gmail.com
#SBATCH --mail-type=ALL

module load pytorch/2.0.1

cmd1="srun python train.py --data_path /pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4 --data_name nskt_16k_sim --in_channels 3 --upscale_factor 4 --model WDSR --lr 0.001 --batch_size 32 --epochs 300"

set -x
bash -c "$cmd1"
