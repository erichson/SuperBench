import re
import json
from collections import defaultdict
import argparse

def parse_file(args):  
    model = args.model
    dataset = args.data_name
    method = args.method
    scale_factor = args.upscale_factor
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

data = parse_file()
with open("parsed_names.json", "w") as outfile:
    json.dump(data, outfile, indent=4)

parser = argparse.ArgumentParser(description='training parameters')
# arguments for data
parser.add_argument('--data_name', type=str, default='cosmo', help='dataset')
parser.add_argument('--data_path', type=str, default='/pscratch/sd/p/puren93/superbench/datasets/nskt16000_1024', help='the folder path of dataset')
parser.add_argument('--method', type=str, default="bicubic", help='downsample method')
parser.add_argument('--crop_size', type=int, default=128, help='crop size for high-resolution snapshots')
parser.add_argument('--n_patches', type=int, default=8, help='number of patches')

# arguments for evaluation
parser.add_argument('--model', type=str, default='shallowDecoder', help='model')
parser.add_argument('--model_path', type=str, default='results/model_EDSR_sst4_0.0001_5544.pt', help='saved model')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--seed', type=int, default=5544, help='random seed')
parser.add_argument('--noise_ratio', type=float, default=0.0, help='noise ratio')

parser.add_argument('--width', type=int, default=1, help='multiply number of channels')
parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
parser.add_argument('--in_channels', type=int, default=1, help='num of input channels')
parser.add_argument('--hidden_channels', type=int, default=32, help='num of hidden channels')
parser.add_argument('--out_channels', type=int, default=1, help='num of output channels')
parser.add_argument('--n_res_blocks', type=int, default=18, help='num of resdiual blocks')

args = parser.parse_args()
print(args)

def load_models_and_testloader(data_name,model_name,upscale_factor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if data_name == "cosmo":
        in_channels = 2
        out_channels = 2
        data_path = '/home/raocp/Desktop/superbench/datasets/cosmo_2048'
    elif data_name == "nskt_16k":
        in_channels = 3
        out_channels = 3
        data_path = '/home/raocp/Desktop/superbench/datasets/nskt16000_1024'
    elif data_name == "nskt_32k":
        in_channels = 3
        out_channels = 3
        data_path = '/home/raocp/Desktop/superbench/datasets/nskt32000_1024'
    elif data_name == "era5":
        in_channels = 3
        out_channels = 3
        data_path = '/home/raocp/Desktop/superbench/datasets/era5'

    resol, n_fields, n_train_samples, mean, std = get_data_info(data_name)
    window_size = 8
    height = (resol[0] // upscale_factor // window_size + 1) * window_size
    width = (resol[1] // upscale_factor // window_size + 1) * window_size
    model_list = {
    'bicubic': Bicubic(upscale_factor=upscale_factor),
    'shallowDecoder': shallowDecoder(resol, upscale_factor=upscale_factor,in_channels=in_channels, out_channels=out_channels),
    'subpixelCNN': subpixelCNN(in_channels, upscale_factor=upscale_factor, width=1),
    'SRCNN': SRCNN(in_channels, upscale_factor),
    'EDSR': EDSR(in_channels, 64, 16, upscale_factor, mean, std),
    'WDSR': WDSR(in_channels, out_channels, 32, 18, upscale_factor, mean, std),
    'SwinIR': SwinIR(upscale=upscale_factor, in_chans=in_channels, img_size=(height, width),
                window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv'),
    'SwinIR_new': SwinIR_new(upscale=upscale_factor, in_chans=in_channels, img_size=(height, width),
                window_size=window_size, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv',mean =mean,std=std),
    }
    model = model_list[model_name]
    model = torch.nn.DataParallel(model)
    if model_name == 'bicubic':
        print('Using bicubic interpolation...')
    else: 
        lr = 0.001 if model_name == "SRCNN" else 0.0001
        if model_name == "SwinIR_new":
            model_path = 'results/model_' + str(model_name) + '_' + str(data_name) + '_' + str(upscale_factor) + '_' + str(lr) + '_' + str(bicubic) +'_' + str(0.0) + '_' + str(args.seed) + '.pt'
        else: 
            model_path = 'results/model_' + str(model_name) + '_' + str(data_name) + '_' + str(upscale_factor) + '_' + str(lr) + '_' + str(bicubic) + '_' + str(5544) + '.pt'
        model = load_checkpoint(model, model_path)
        model = model.to(device)  
        
    return model,testloader
