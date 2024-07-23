import glob
DATA_INFO = {"nskt_16k": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_16k",3],
             "nskt_32k": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_32k",3],
            "nskt_16k_sim_4_v7": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4_v7",3],
            "nskt_32k_sim_4_v7": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4_v7",3],
            # "nskt_16k_sim_4": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_4",3],
            # "nskt_32k_sim_4": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_4",3],
            # "nskt_16k_sim_2": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_16k_sim_2",3],
            # "nskt_32k_sim_2": ["/pscratch/sd/j/junyi012/superbench_v2/nskt_32k_sim_2",3],
            "cosmo": ["/pscratch/sd/j/junyi012/superbench_v2/cosmo_v2",2],
            "cosmo_sim_8":["/pscratch/sd/j/junyi012/superbench_v2/cosmo_lre_sim_s8_v2",2],
              }

MODEL_INFO = {"SRCNN": {"lr": 1e-3,"batch_size": 64,"epochs": 300},
            "subpixelCNN": {"lr": 1e-3,"batch_size": 64,"epochs": 300},
            "EDSR": {"lr": 1e-3,"batch_size": 32,"epochs": 300},
            "WDSR": {"lr": 1e-4,"batch_size": 32,"epochs": 300},
            "SwinIR": {"lr": 1e-4,"batch_size": 32,"epochs":300},
            "FNO2D": {"lr": 1e-3,"batch_size": 32,"epochs": 500},}

def find_model_path(data_name, model, scale_factor, method =None, tag=None):
    pattern = f"results/model_{model}_{data_name}_{scale_factor}_*"
    if method is not None:
        pattern += f"_{method}"
    pattern += "*.pt"
    files = glob.glob(pattern)
    if files:
        return files  # Returns a list of matching files
    else:
        print(f"No matching model found (pattern: {pattern})")
        return None

# Example usage

# list = ["cosmo","FNO2D","16","50"]
# list += ["cosmo","FNO2D","8","852"]
for data_name in ["cosmo"]:
    for model in ["FNO2D","SwinIR","WDSR","EDSR","subpixelCNN","SRCNN","Bicubic"]:  
        for scale_factor in [8,16]:
            model_paths = find_model_path(data_name, model, scale_factor,"bicubic")
            if model_paths is not None:
                for path in model_paths:
                    with open ("temp_eval.sh","a") as f:
                        print(f"python eval.py --data_name {data_name} --upscale_factor {scale_factor} --model {model} --data_path {DATA_INFO[data_name][0]} --in_channels {DATA_INFO[data_name][1]} --model_path {path} ",file=f)
