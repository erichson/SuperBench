import numpy as np
import torch
from torch import nn
import argparse
from tqdm import tqdm

from src.get_data import getData
from src.models import *


parser = argparse.ArgumentParser(description='training parameters')
parser.add_argument('--data', type=str, default='DoubleGyre', help='dataset')
parser.add_argument('--model', type=str, default='shallowDecoder', help='model')
parser.add_argument('--epochs', type=int, default=300, help='max epochs')
parser.add_argument('--device', type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help='computing device')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--wd', type=float, default=0, help='weight decay')
parser.add_argument('--seed', type=int, default=5544, help='random seed')

parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')


args = parser.parse_args()
print(args)

#==============================================================================
# Set random seed to reproduce the work
#==============================================================================
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


#******************************************************************************
# Get data
#******************************************************************************
train_loader, test1_loader, val1_loader, test2_loader, val2_loader = getData(args.data, train_bs=args.batch_size)

for inp, label in test2_loader:
    print('{}:{}'.format(inp.shape, label.shape,))
    break

#==============================================================================
# Get model
#==============================================================================
if args.data == 'isoflow':
    input_size = [64, 64] 
    output_size = [256, 256]
elif args.data == 'DoubleGyre':
    input_size = [112, 48] 
    output_size = [448, 192]
elif args.data == 'RBC':
    input_size = [32, 32] 
    output_size = [256, 256]    
    
model_list = {
        'shallowDecoder': shallowDecoder(input_size, output_size),
        'shallowDecoderV2': shallowDecoderV2(input_size, output_size),
        'subpixelCNN': subpixelCNN(upscale_factor=args.upscale_factor)
}

model = model_list[args.model].to(args.device)
model = torch.nn.DataParallel(model)

#==============================================================================
# Model summary
#==============================================================================
print(model)    
print('**** Setup ****')
print('Total params Generator: %.7fM' % (sum(p.numel() for p in model.parameters())/1000000.0))
print('************')    


#******************************************************************************
# Optimizer, Loss Function and Learning Rate Scheduler
#******************************************************************************
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
criterion = nn.MSELoss().to(args.device)


def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

scheduler = torch.optim.lr_scheduler.LambdaLR(
          optimizer,
          lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
              step, args.epochs * len(train_loader),
              1,  # lr_lambda computes multiplicative factor
              1e-6 / args.lr))    

#******************************************************************************
# Validate
#******************************************************************************

def validate(val1_loader, val2_loader, model):
    for batch_idx, (data, target) in enumerate(val1_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        mse1 = criterion(output, target)

    for batch_idx, (data, target) in enumerate(val2_loader):
        data, target = data.to(args.device).float(), target.to(args.device).float()
        output = model(data) 
        mse2 = criterion(output, target)


    return mse1.item(), mse2.item()

#******************************************************************************
# Start training
#******************************************************************************
best_val = np.inf
for epoch in range(args.epochs):
    
    #for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
    for batch_idx, (data, target) in enumerate(train_loader):
        
        data, target = data.to(args.device).float(), target.to(args.device).float()

        # ===================forward=====================
        model.train()
        output = model(data) 
        loss = criterion(output, target)
        
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()


    # =============== validate ======================
    mse1, mse2 = validate(val1_loader, val2_loader, model)
    print("epoch: %s, val1 error: %.10f, val2 error: %.10f" % (epoch, mse1, mse2))      
            

    if (mse1+mse2) <= best_val:
        best_val = mse1+mse2
        torch.save(model, 'results/model_' + str(args.model) + '_' + str(args.data) + '_' + str(args.lr) + '_' + str(args.seed) + '.npy' )


