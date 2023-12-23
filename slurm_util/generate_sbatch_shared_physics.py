import argparse


DATA_INFO = {"nskt_16k": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_16k",3],
             "nskt_32k": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_32k",3],
            "nskt_16k_sim_4_v7": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7",3],
            "nskt_32k_sim_4_v7": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7",3],
            "cosmo": ["/pscratch/sd/j/junyi012/superbench_v2/cosmo2048",2],
            }

MODEL_INFO = {"SRCNN": {"lr": 1e-3,"batch_size": 64,"epochs": 300},
            "subpixelCNN": {"lr": 1e-3,"batch_size": 64,"epochs": 300},
            "EDSR": {"lr": 1e-3,"batch_size": 32,"epochs": 300},
            "WDSR": {"lr": 1e-4,"batch_size": 32,"epochs": 300},
            "SwinIR": {"lr": 1e-4,"batch_size": 32,"epochs":300},
            "FNO2D": {"lr": 1e-3,"batch_size": 32,"epochs": 500},}

def generate_bash_script(data_name, model_name, scale_factor, downsample_method="bicubic", noise=0.0,lamb_p=0.0):
    job_name = f"{data_name}_{model_name}_{scale_factor}_{lamb_p}"


    bash_content = f"""#!/bin/bash
#SBATCH --time=18:00:00
#SBATCH --account=dasrepo_g
#SBATCH --job-name={job_name}
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

cmd1="srun python train_fluid.py --data_path {DATA_INFO[data_name][0]} --data_name {data_name} --in_channels {DATA_INFO[data_name][1]} --upscale_factor {scale_factor} --model {model_name} --lr {MODEL_INFO[model_name]['lr']} --batch_size {MODEL_INFO[model_name]['batch_size']} --epochs {MODEL_INFO[model_name]['epochs']} --noise_ratio {noise} --method {downsample_method} --phy_loss_weight {lamb_p}"

set -x
bash -c "$cmd1"
"""

    with open(f"make_file/{job_name}.sbatch", 'w') as out_file:
        out_file.write(bash_content)
        print(f"Bash script generated as {job_name}.sbatch")
    return  job_name
# Run the function
if __name__ == "__main__":
    # data_name_list = ["cosmo"]
    data_name_list = ["nskt_16k_sim_4_v7","nskt_32k_sim_4_v7"]
    # model_name_list =  ["SRCNN","WDSR","SwinIR","subpixelCNN","EDSR"]
    model_name_list = ["SwinIR"]
    downsample_method = ["bicubic"]
    lamb = [0.001]
    # for name in data_name_list:
    #     for scale_factor in [8,16]:
    #         for model_name in model_name_list:
    #             job_name = generate_bash_script(data_name=name,model_name=model_name,scale_factor=scale_factor)
    #             with open("bash2slurm.sh","a") as f:
    #                 print(f"sbatch make_file/{job_name}.sbatch",file=f)
    #             f.close()
    for name in data_name_list:
        for lamb_p in lamb:
            for s in [4]:
                for model_name in model_name_list:
                    job_name = generate_bash_script(data_name=name,model_name=model_name,scale_factor=s,lamb_p=lamb_p)
                    with open("bash2slurm.sh","a") as f:
                        print(f"sbatch make_file/{job_name}.sbatch",file=f)
                    f.close()
