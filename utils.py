import numpy as np
import torch
from torch import nn

def get_data_info(data_name):
    if data_name == 'nskt_16k':
        resol = [1024, 1024] 
        n_fields = 3
        n_train_samples = 1200
        mean = [-0.0218,  0.0309,  0.1429] 
        std = [0.6703, 0.6344, 8.3615]

    elif data_name == 'nskt_32k':
        resol = [1024, 1024]
        n_fields = 3
        n_train_samples = 1200
        mean = [ 0.0079,  0.0463, -0.0219] 
        std = [0.6652, 0.6485, 8.9646]

    elif data_name == 'cosmo':
        resol = [2048, 2048] 
        n_fields = 2
        n_train_samples = 1200
        mean = [ 3.9017, -0.3575] # [ 3.8956, -0.3664] 
        std = [0.2266, 0.4048] # [0.2191, 0.3994]

    elif data_name == 'cosmo_lres_sim':
        resol = [4096, 4096] 
        n_fields = 2
        n_train_samples = 400
        mean = [3.8990, -0.3613] 
        std = [0.2237, 0.4039]  

    elif data_name == 'era5':
        resol = [720, 1440]
        n_fields = 3
        n_train_samples = 6*365
        mean = [6.3024, 278.3945, 18.4262] 
        std = [3.7376, 21.0588, 16.4687]
    else:
        raise ValueError('dataset {} not recognized'.format(data_name))

    return resol, n_fields, n_train_samples, mean, std


def set_optimizer(args, model):
    if args.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.optimizer_type == 'AdamW':
        # swin transformer
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    else:
        raise ValueError('Optimizer type {} not recognized'.format(args.optimizer_type))
    return optimizer


def get_lr(step, total_steps, lr_max, lr_min):
    """Compute learning rate according to cosine annealing schedule."""
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def set_scheduler(args, optimizer, train_loader):
    if args.scheduler_type == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                    step, args.epochs * len(train_loader),
                    1,  # lr_lambda computes multiplicative factor
                    1e-6 / args.lr))  

    elif args.scheduler_type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.gamma)

    elif args.scheduler_type == 'ExponentialLR':
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.gamma)

    return scheduler


def loss_function(args):
    if args.loss_type == 'l1':
        print('Training with L1 loss...')
        criterion = nn.L1Loss().to(args.device)
    elif args.loss_type == 'l2': 
        print('Training with L2 loss...')
        criterion = nn.MSELoss().to(args.device)
    else:
        raise ValueError('Loss type {} not recognized'.format(args.loss_type))
    return criterion

def save_checkpoint(model, optimizer,save_path):
    '''save model and optimizer'''
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
        }, save_path)


def load_checkpoint(model, save_path):
    '''load model and optimizer'''
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Model loaded...')

    return model
