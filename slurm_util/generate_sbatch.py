import argparse


DATA_INFO = {"nskt_16k": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_16k",3],
             "nskt_32k": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_32k",3],
            "nskt_16k_sim": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4",3],
            "nskt_32k_sim": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4",3],
            "cosmo": ["/pscratch/sd/j/junyi012/superbench_v2/cosmo2048",2],
              }

MODEL_INFO = {"SRCNN": {"lr": 1e-3,"batch_size": 64,"epochs": 300},
            "subpixelCNN": {"lr": 1e-3,"batch_size": 64,"epochs": 300},
            "EDSR": {"lr": 1e-3,"batch_size": 32,"epochs": 300},
            "WDSR": {"lr": 1e-4,"batch_size": 32,"epochs": 300},
            "SwinIR": {"lr": 1e-4,"batch_size": 32,"epochs":300},
            "FNO2D": {"lr": 1e-3,"batch_size": 32,"epochs": 500},}

def generate_bash_script(data_name, model_name, scale_factor):
    job_name = f"{data_name}_{model_name}_{scale_factor}"

    if "FNO" in model_name:
        file = "train_FNO.py"
    else:
        file = "train.py"

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

cmd1="srun python {file} --data_path {DATA_INFO[data_name][0]} --data_name {data_name} --in_channels {DATA_INFO[data_name][1]} --upscale_factor {scale_factor} --model {model_name} --lr {MODEL_INFO[model_name]['lr']} --batch_size {MODEL_INFO[model_name]['batch_size']} --epochs {MODEL_INFO[model_name]['epochs']}"

set -x
bash -c "$cmd1"
"""

    with open(f"make_file/{job_name}.sbatch", 'w') as out_file:
        out_file.write(bash_content)
        print(f"Bash script generated as {job_name}.sbatch")
    return  job_name
# Run the function
if __name__ == "__main__":
    data_name_list = ["cosmo"]
    model_name_list =  ["FNO2D","WDSR"]
    for name in data_name_list:
        for scale_factor in [8,16]:
            for model_name in model_name_list:
                job_name = generate_bash_script(data_name=name,model_name=model_name,scale_factor=scale_factor)
                with open("bash2slurm.sh","a") as f:
                    print(f"sbatch make_file/{job_name}.sbatch",file=f)
                f.close()
    # for name in ["nskt_16k_sim","nskt_32k_sim"]:
    #     for scale_factor in [4]:
    #         for model_name in model_name_list:
    #             job_name = generate_bash_script(data_name=name,model_name=model_name,scale_factor=scale_factor)
    #             with open("bash2slurm.sh","a") as f:
    #                 print(f"sbatch make_file/{job_name}.sbatch",file=f)
    #             f.close()
