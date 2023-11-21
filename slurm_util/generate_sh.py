import os
import itertools
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='SuperBench Training File Generator')
parser.add_argument('--model', default='SRCNN', required=False, help='model name:[SRCNN, subpixelCNN, EDSR, WDSR, SwinIR]')
parser.add_argument('--data', default='nskt_16k', required=False, help='data name:[nskt_16k, nskt_32k, cosmo_lres_sim, era5]')
parser.add_argument('--gpus', nargs='+', type=int, default=[0,1,2,3], help="ids of gpus to use")
parser.add_argument('--generate_all', action="store_true", default=False, help='If set true, generate all training scripts in one .sh file')

args = parser.parse_args()
generate_all = args.generate_all
if not generate_all:
    model_name = args.model
    data_name = args.data
gpu_list = args.gpus

training_param_fp = './model_data_training_param.csv'
output_dir = '../'

training_params = pd.read_csv(training_param_fp)
if not generate_all:
    training_params = training_params[(training_params['model_name']==model_name) & (training_params['data_name']==data_name)]
else:
    training_params = training_params
training_params_list = training_params.to_dict('records')
training_params_with_method = []

for training_param in training_params_list:
    if training_param['data_name'] == 'cosmo_lres_sim':
        new_training_param = {}
        new_training_param['upscale_factor'] = 8
        new_training_param.update(training_param)
        training_params_with_method.append(new_training_param)
    else:
        for u_factor, u_method in list(itertools.product([8, 16], ['bicubic', 'uniform_noise'])):
            if u_method == 'bicubic':
                new_training_param = {}
                new_training_param['method'] = u_method
                new_training_param['upscale_factor'] = u_factor
                new_training_param.update(training_param)
                training_params_with_method.append(new_training_param)
            elif u_factor == 8:
                for n_ratio in [0.05, 0.1]:
                    new_training_param = {}
                    new_training_param['method'] = u_method
                    new_training_param['noise_ratio'] = n_ratio
                    new_training_param['upscale_factor'] = u_factor
                    new_training_param.update(training_param)
                    training_params_with_method.append(new_training_param)

sh_cmds = []

gpu_str = ''
for gpu_id in gpu_list:
    gpu_str += str(gpu_id) + ','
gpu_str = gpu_str[:-1]

sh_cmds.append(f'export CUDA_VISIBLE_DEVICES={gpu_str}\n')

for training_param in training_params_with_method:
    cmds = 'python train.py \\\n'
    for ind, (k, v) in enumerate(training_param.items()):
        cmd = f'    --{k} {v} \\\n'
        if k == 'model_name':
            cmd = f'    --model {v} \\\n'
        elif k in ['data_path', 'noise_ratio', 'method']:
            cmd = f"    --{k} '{v}' \\\n"
        if ind == len(training_param)-1:
            cmd = cmd[:-2] + ' &\n\n'
        cmds += cmd        
    sh_cmds.append(cmds)

    sh_cmds.append('pid_file6=$!\n')
    sh_cmds.append('echo "PID2 for train.py: $pid_file6" >> pid.log\n')
    sh_cmds.append('wait $pid_file6\n\n')

output_file_name = 'run_train_all.sh' if generate_all else f'run_train_{model_name}_{data_name}.sh'
with(open(os.path.join(output_dir, output_file_name), 'w')) as f:
    f.writelines(sh_cmds)