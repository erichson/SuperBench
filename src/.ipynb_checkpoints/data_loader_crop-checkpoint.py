import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import h5py
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import torchvision.transforms.functional as F


def getData(args, n_patches, std):  
    '''
    Loading data from four dataset folders: (a) nskt_16k; (b) nskt_32k; (c) cosmo; (d) era5.
    Each dataset contains: 
        - 1 train dataset, 
        - 2 validation sets (interpolation and extrapolation), 
        - 2 test sets (interpolation and extrapolation)
        
    ===
    std: the channel-wise standard deviation of each dataset, list: [#channels]
    '''

    train_loader = get_data_loader(args, args.data_path+'/train', train=True, n_patches=n_patches, std=std)
    val1_loader = get_data_loader(args, args.data_path+'/valid_1', train=True, n_patches=n_patches, std=std)        
    val2_loader = get_data_loader(args, args.data_path+'/valid_2', train=True, n_patches=n_patches, std=std)         
    test1_loader = get_data_loader(args, args.data_path+'/test_1', train=False, n_patches=n_patches, std=std)        
    test2_loader = get_data_loader(args, args.data_path+'/test_2', train=False, n_patches=n_patches, std=std)
        
    return train_loader, val1_loader, val2_loader, test1_loader, test2_loader 


def get_data_loader(args, data_path, train, n_patches, std):
    
    transform = torch.from_numpy

    if args.data_name == 'nskt_16k' or args.data_name == 'nskt_32k' or args.data_name == 'cosmo':
        dataset = GetFluidDataset(data_path, train, transform, args.upscale_factor, args.noise_ratio, std, args.crop_size,n_patches,args.method) 

    elif args.data_name == 'era5':
        dataset = GetClimateDataset(data_path, train, transform, args.upscale_factor, args.noise_ratio, std, args.crop_size,n_patches,args.method) 
    
    else:
        raise ValueError('dataset {} not recognized'.format(args.data_name))

    dataloader = DataLoader(dataset,
                            batch_size = int(args.batch_size),
                            num_workers = 4, # TODO: make a param
                            shuffle = (train == True), # TODO: now validation set will also shuffle. If need change here, a new variable validation shoud be added.   
                            sampler = None,
                            drop_last = False,
                            pin_memory = torch.cuda.is_available())

    return dataloader


class GetFluidDataset(Dataset):
    '''Dataloader class for NSKT and cosmo datasets'''
    def __init__(self, location, train, transform, upscale_factor, noise_ratio, std,crop_size, n_patches, method):
        self.location = location
        self.upscale_factor = upscale_factor
        self.train = train
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self._get_files_stats()
        self.crop_size = crop_size
        self.n_patches = n_patches
        self.crop_transform = transforms.RandomCrop(crop_size)
        self.method = method
        if (train == True) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int(self.crop_size/upscale_factor),int(self.crop_size/upscale_factor)),Image.BICUBIC)  # subsampling the image (half size)
        elif (train == False) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int(self.img_shape_x/upscale_factor),int(self.img_shape_y/upscale_factor)),Image.BICUBIC)  # subsampling the image (half size)


    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_files = len(self.files_paths)
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_file = _f['fields'].shape[0]
            self.n_in_channels = _f['fields'].shape[1]
            self.img_shape_x = _f['fields'].shape[2]
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_files * self.n_samples_per_file
        self.files = [None for _ in range(self.n_files)]
        print("Number of samples per file: {}".format(self.n_samples_per_file))
        print("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['fields']  

    def __len__(self):
        if self.train == True:
            return self.n_samples_total*self.n_patches
        else:
            return self.n_samples_total

    def __getitem__(self, global_idx):

        file_idx, local_idx = self.get_indices(global_idx) 
        #open image file
        if self.files[file_idx] is None:
            self._open_file(file_idx)
        # for NSKT and cosmo, the loaded high-res data are in numpy tensor, [channel, height, width]  
        # Apply transform 
        y = self.transform(self.files[file_idx][local_idx])
        if self.train:
            y = self.crop_transform(y)
        X = self.get_X(y)

        return X, y

    def get_indices(self, global_idx):
        if self.train:
            file_idx = int(global_idx/(self.n_samples_per_file*self.n_patches))  # which file we are on
            local_idx = int((global_idx//self.n_patches) % self.n_samples_per_file)  # which sample in that file we are on 
        else:
            file_idx = int(global_idx/self.n_samples_per_file)  # which file we are on
            local_idx = int(global_idx % self.n_samples_per_file)  # which sample in that file we are on 

        return file_idx, local_idx

    def get_X(self, y):
        if self.method == "uniform":
            X = y[:, ::self.upscale_factor, ::self.upscale_factor]
        elif self.method == "noisy_uniform":
            X = y[:, ::self.upscale_factor, ::self.upscale_factor]
            X = X + self.noise_ratio * self.std * torch.randn(X.shape)
        elif self.method == "bicubic":
            X = self.bicubicDown_transform(y)
        else:
            raise ValueError(f"Invalid method: {self.method}")
        #TODO: add gaussian blur
        return X


class GetClimateDataset(Dataset):
    '''Dataloader class for climate datasets'''
    def __init__(self, location, train, transform, upscale_factor, noise_ratio, std, crop_size,n_patches,method):
        self.location = location
        self.upscale_factor = upscale_factor
        self.train = train
        self.noise_ratio = noise_ratio
        self.std = torch.Tensor(std).view(len(std),1,1)
        self.transform = transform
        self._get_files_stats()
        self.crop_size = crop_size
        self.n_patches = n_patches
        self.method = method
        self.crop_transform = transforms.RandomCrop(crop_size)
        # we will always crop the image into square patches
        if (self.train == True) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int(self.crop_size/upscale_factor),int(self.crop_size/upscale_factor)),Image.BICUBIC)  # subsampling the image (half size)
        elif (self.train == False) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int((self.img_shape_x-1)/upscale_factor),int(self.img_shape_y/upscale_factor)),Image.BICUBIC)  # subsampling the image (half size)

    def _get_files_stats(self):
        self.files_paths = glob.glob(self.location + "/*.h5")
        self.files_paths.sort()
        self.n_years = len(self.files_paths)
        with h5py.File(self.files_paths[0], 'r') as _f:
            print("Getting file stats from {}".format(self.files_paths[0]))
            self.n_samples_per_year = _f['fields'].shape[0]
            self.n_in_channels = _f['fields'].shape[1]
            self.img_shape_x = _f['fields'].shape[2]
            self.img_shape_y = _f['fields'].shape[3]

        self.n_samples_total = self.n_years * self.n_samples_per_year
        self.files = [None for _ in range(self.n_years)]
        print("Number of samples per year: {}".format(self.n_samples_per_year))
        print("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))

    def _open_file(self, year_idx):
        _file = h5py.File(self.files_paths[year_idx], 'r')
        self.files[year_idx] = _file['fields']  

    def __len__(self):
        if self.train == True: 
            return self.n_samples_total * self.n_patches
        return self.n_samples_total

    def __getitem__(self, global_idx):
        year_idx, local_idx = self.get_indices(global_idx)

        # Open image file if it's not already open
        if self.files[year_idx] is None:
            self._open_file(year_idx)

        # Apply transform and cut-off
        y = self.transform(self.files[year_idx][local_idx])
        y = y[:,:-1,:]

        # Modify y for training and get X based on method
        if self.train:
            y = self.crop_transform(y)
        X = self.get_X(y)

        return X, y

    def get_indices(self, global_idx):
        if self.train:
            year_idx = int(global_idx/(self.n_samples_per_year*self.n_patches))  # which year we are on
            local_idx = int((global_idx//self.n_patches) % self.n_samples_per_year)  # which sample in that year we are on 
        else:
            year_idx = int(global_idx/self.n_samples_per_year)  # which year we are on
            local_idx = int(global_idx % self.n_samples_per_year)  # which sample in that year we are on 

        return year_idx, local_idx

    def get_X(self, y):
        if self.method == "uniform":
            X = y[:, ::self.upscale_factor, ::self.upscale_factor]
        elif self.method == "noisy_uniform":
            X = y[:, ::self.upscale_factor, ::self.upscale_factor]
            X = X + self.noise_ratio * self.std * torch.randn(X.shape)
        elif self.method == "bicubic":
            X = self.bicubicDown_transform(y)
        else:
            raise ValueError(f"Invalid method: {self.method}")

        return X
