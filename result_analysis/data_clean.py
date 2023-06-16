import re
import json
from collections import defaultdict



def parse_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.read().split('\n')

    data = []

    for line in lines:
        if line.startswith(' model'):
            # Split the line by empty spaces
            fields = re.split(r'\s+', line)
            
            model = fields[1].replace('model', '').replace('SwinIR_new', 'SwinIR').replace('bicubic','Bicubic')
            dataset = fields[3]
            method = fields[5]
            scale_factor = fields[8]
            noise_ratio = fields[11]

            model_info = {
                "model": model,
                "dataset": dataset,
                "method": method,
                "scale factor": scale_factor,
                "noise ratio": noise_ratio,
                "metrics": {},
            }

            data.append(model_info)

        elif line.startswith('PSNR') or line.startswith('SSIM'):
            fields = re.split(r'\s+', line)
            metric = fields[0]
            test1_error = fields[4].replace(',', '')
            test2_error = fields[7]
            data[-1]["metrics"][metric] = {
                "test1 error": test1_error,
                "test2 error": test2_error,
            }
        elif line.startswith('Infinity norm'):
            fields = re.split(r'\s+', line)
            metric = fields[0]
            test1_error = fields[5].replace(',', '')
            test2_error = fields[8]
            data[-1]["metrics"][metric] = {
                "test1 error": test1_error,
                "test2 error": test2_error,
            }
        elif line.startswith('RFNE'):
            fields = re.split(r'\s+', line)
            metric = fields[0]
            test1_error = fields[4].replace(',', '')
            test2_error = fields[8]
            data[-1]["metrics"][metric] = {
                "test1 error": test1_error,
                "test2 error": test2_error,
            }
        elif line.startswith('Physics loss'):
            fields = re.split(r'\s+', line)
            metric = fields[0]
            test1_error = fields[5].replace(',', '')
            test2_error = fields[8]
            data[-1]["metrics"][metric] = {
                "test1 error": test1_error,
                "test2 error": test2_error,
            }
    # First, group the dictionaries by their 'model', 'dataset', 'method', 'scale factor', and 'noise ratio' fields.
    groups = defaultdict(list)
    for entry in data:
        key = (entry['model'], entry['dataset'], entry['method'], entry['scale factor'], entry['noise ratio'])
        groups[key].append(entry)

    # Then, for each group, merge the 'metrics' dictionaries.
    merged_data = []
    for key, group in groups.items():
        merged_metrics = {}
        for entry in group:
            merged_metrics.update(entry['metrics'])

        merged_entry = {
            "model": key[0],
            "dataset": key[1],
            "method": key[2],
            "scale factor": key[3],
            "noise ratio": key[4],
            "metrics": merged_metrics
        }
        merged_data.append(merged_entry)
    return merged_data

file_path = 'result_ver2.txt'
data = parse_file(file_path)
with open("normed_eval.json", "w") as outfile:
    json.dump(data, outfile, indent=4)

with open('normed_eval.json', 'r') as f:
    data = json.load(f)

print(json.dumps(data, indent=4))


# model_info = {
#     "model": model,
#     "dataset": dataset,
#     "method": method,
#     "scale factor": scale_factor,
#     "noise ratio": noise_ratio,
#     "metrics": {},
# }
models = ['Bicubic', 'SRCNN', 'subpixelCNN', 'EDSR', 'WDSR', 'SwinIR']
metrics_list = ['RFNE', 'Infinity', 'PSNR', 'SSIM', 'Physics']

rfne8, inf_norm8, psnr8, ssim8 = [], [], [], []
rfne16, inf_norm16, psnr16, ssim16 = [], [], [], []
for model_name in models:
    for entry in data: 
       if entry['model'] == model_name:
            if entry['dataset'] == 'era5' and entry['method'] == 'noisy_uniform':
                if entry['scale factor'] == '8':
                    rfne8.append(entry['metrics']['RFNE']['test2 error'])
                    inf_norm8.append(entry['metrics'].get('Infinity', {}).get('test2 error', None))
                    psnr8.append(entry['metrics'].get('PSNR', {}).get('test2 error', None))
                    ssim8.append(entry['metrics'].get('SSIM', {}).get('test2 error', None))
                elif entry['scale factor'] == '16':
                    rfne16.append(entry['metrics'].get('RFNE', {}).get('test2 error', None))
                    inf_norm16.append(entry['metrics'].get('Infinity', {}).get('test2 error', None))
                    psnr16.append(entry['metrics'].get('PSNR', {}).get('test2 error', None))
                    ssim16.append(entry['metrics'].get('SSIM', {}).get('test2 error', None))

print("RFNE 8: ", rfne8)
print("Infinity norm 8: ", inf_norm8)
print("PSNR 8: ", psnr8)
print("SSIM 8: ", ssim8)

print("RFNE 16: ", rfne16)
print("Infinity norm 16: ", inf_norm16)
print("PSNR 16: ", psnr16)
print("SSIM 16: ", ssim16) 
