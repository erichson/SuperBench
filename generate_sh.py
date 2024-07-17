import argparse


DATA_INFO = {"nskt_16k": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_16k",3],
             "nskt_32k": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_32k",3],
            "nskt_16k_sim_4_v7": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7",3],
            "nskt_32k_sim_4_v7": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7",3],
            "cosmo": ["/pscratch/sd/j/junyi012/superbench_v2/cosmo2048",2],
            "cosmo_sim_8":["/pscratch/sd/j/junyi012/superbench_v2/cosmo_lre_sim_s8_v2",2],
            "era5": ["/pscratch/sd/j/junyi012/superbench_v2/era5",3],
              }

MODEL_INFO = {"SRCNN": {"lr": 4e-3,"batch_size": 256,"epochs": 300},
            "subpixelCNN": {"lr": 4e-3,"batch_size": 256,"epochs": 300},
            "EDSR": {"lr": 4e-3,"batch_size": 256,"epochs": 300},
            "WDSR": {"lr": 8e-4,"batch_size": 256,"epochs": 300},
            "SwinIR": {"lr": 8e-4,"batch_size": 256,"epochs":400},
            "FNO2D": {"lr": 1e-3,"batch_size": 256,"epochs": 500},}

def generate_bash_script(data_name, model_name, scale_factor,num_pathches=8,crop_size=128):
    job_name = f"{data_name}_{model_name}_{scale_factor}"

    if "FNO" in model_name:
        file = "train_FNO.py"
    else:
        file = "train.py"

    bash_content = f"python {file} --data_path {DATA_INFO[data_name][0]} --data_name {data_name} --in_channels {DATA_INFO[data_name][1]} --upscale_factor {scale_factor} --model {model_name} --lr {MODEL_INFO[model_name]['lr']} --batch_size {MODEL_INFO[model_name]['batch_size']} --epochs {MODEL_INFO[model_name]['epochs']} --n_patches {num_pathches} --crop_size {crop_size};"
    return bash_content

def generate_bash_script_fluid(data_name, model_name, scale_factor, downsample_method="bicubic", noise=0.0,lamb_p=0.001):
    job_name = f"{data_name}_{model_name}_{scale_factor}_{lamb_p}"
    bash_content = f"python train_fluid.py --data_path {DATA_INFO[data_name][0]} --data_name {data_name} --in_channels {DATA_INFO[data_name][1]} --upscale_factor {scale_factor} --model {model_name} --lr {MODEL_INFO[model_name]['lr']} --batch_size {MODEL_INFO[model_name]['batch_size']} --epochs {MODEL_INFO[model_name]['epochs']} --noise_ratio {noise} --method {downsample_method} --phy_loss_weight {lamb_p};"
    return bash_content

# Run the function
if __name__ == "__main__":
    # data_name_list = ["cosmo"]
    data_name_list = ["nskt_16k","nskt_32k","cosmo","era5"]
    model_name_list =  ["SRCNN","subpixelCNN","EDSR","FNO2D","WDSR","SwinIR"]
    # model_name_list =  []
    for name in data_name_list:
        for scale_factor in [8,16]:
            for model_name in model_name_list:
                job_name = generate_bash_script(data_name=name,model_name=model_name,scale_factor=scale_factor,num_pathches=8)
                with open("train_all.sh","a") as f:
                    print(f"{job_name}",file=f)
                f.close()
                if name.startswith("nskt") and model_name.startswith("SwinIR"):
                    job_name = generate_bash_script_fluid(data_name=name,model_name=model_name,scale_factor=scale_factor)
                    with open("train_all.sh","a") as f:
                        print(f"{job_name}",file=f)
                    f.close()

    data_name_list = ["nskt_16k_sim_4_v7","nskt_32k_sim_4_v7"]
    for name in data_name_list:
        for scale_factor in [4]:
            for model_name in model_name_list:
                job_name = generate_bash_script(data_name=name,model_name=model_name,scale_factor=scale_factor,num_pathches=8)
                with open("train_all.sh","a") as f:
                    print(f"sbatch make_file/{job_name}.sh",file=f)
                f.close()
                if name.startswith("nskt") and model_name.startswith("SwinIR"):
                    job_name = generate_bash_script_fluid(data_name=name,model_name=model_name,scale_factor=scale_factor,downsample_method="lr_sim")
                    with open("train_all.sh","a") as f:
                        print(f"sbatch make_file/{job_name}.sh",file=f)
                    f.close()

    data_name_list = ["cosmo_sim_8"]
    for name in data_name_list:
        for scale_factor in [8]:
            for model_name in model_name_list:
                job_name = generate_bash_script(data_name=name,model_name=model_name,scale_factor=scale_factor,num_pathches=8)
                with open("train_all.sh","a") as f:
                    print(f"sbatch make_file/{job_name}.sh",file=f)
                f.close()
