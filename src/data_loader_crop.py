import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import h5py
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

def getData(args, n_patches, std,patched_eval=False,test=False):  
    '''
    Loading data from four dataset folders: (a) nskt_16k; (b) nskt_32k; (c) cosmo; (d) era5.
    Each dataset contains: 
        - 1 train dataset, 
        - 2 validation sets (interpolation and extrapolation), 
        - 2 test sets (interpolation and extrapolation)
        
    ===
    std: the channel-wise standard deviation of each dataset, list: [#channels]
    '''
    if test == True:
        test1_loader = get_data_loader(args, '/test_1', train=patched_eval, n_patches=n_patches, std=std)        
        test2_loader = get_data_loader(args, '/test_2', train=patched_eval, n_patches=n_patches, std=std)
        return test1_loader, test2_loader
    else:
        train_loader = get_data_loader(args, '/train', train=True, n_patches=n_patches, std=std)
        val1_loader = get_data_loader(args, '/valid_1', train=True, n_patches=n_patches, std=std)        
        val2_loader = get_data_loader(args, '/valid_2', train=True, n_patches=n_patches, std=std)         
        test1_loader = get_data_loader(args, '/test_1', train=patched_eval, n_patches=n_patches, std=std)        
        test2_loader = get_data_loader(args, '/test_2', train=patched_eval, n_patches=n_patches, std=std)
            
        return train_loader, val1_loader, val2_loader, test1_loader, test2_loader 


def get_data_loader(args, data_tag, train, n_patches, std):
    
    transform = torch.from_numpy

    if args.data_name == 'nskt_16k' or args.data_name == 'nskt_32k' or args.data_name == 'cosmo':
        dataset = GetFluidDataset(args.data_path+data_tag, train, transform, args.upscale_factor, args.noise_ratio, std, args.crop_size,n_patches,args.method) 

    elif args.data_name == 'era5':
        dataset = GetClimateDataset(args.data_path+data_tag, train, transform, args.upscale_factor, args.noise_ratio, std, args.crop_size,n_patches,args.method) 
    
    elif args.data_name == 'cosmo_lres_sim':
        # print('Using low-resolution simulation degradation...')
        dataset = GetCosmoSimData(args.data_path+data_tag, data_tag, train, transform, args.crop_size, n_patches)
    elif args.data_name.startswith("nskt_16k_sim") or args.data_name.startswith("nskt_32k_sim") or args.data_name.startswith("cosmo_sim"):
        dataset = GetFluidDataset_LRsim(args.data_path+data_tag, train, transform, args.upscale_factor, args.noise_ratio, std, args.crop_size,n_patches,args.method) 

    else:
        raise ValueError('dataset {} not recognized'.format(args.data_name))

    dataloader = DataLoader(dataset,
                            batch_size = int(args.batch_size),
                            num_workers = 2, # TODO: make a param
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
            self.bicubicDown_transform = transforms.Resize((int(self.crop_size/upscale_factor),int(self.crop_size/upscale_factor)),Image.BICUBIC,antialias=True)  # subsampling the image (half size)
        elif (train == False) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int(self.img_shape_x/upscale_factor),int(self.img_shape_y/upscale_factor)),Image.BICUBIC,antialias=True)  # subsampling the image (half size)


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



class GetFluidDataset_LRsim(Dataset):
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
            self.bicubicDown_transform = transforms.Resize((int(self.crop_size/upscale_factor),int(self.crop_size/upscale_factor)),Image.BICUBIC,antialias=True)  # subsampling the image (half size)
        elif (train == False) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int(self.img_shape_x/upscale_factor),int(self.img_shape_y/upscale_factor)),Image.BICUBIC,antialias=True)  # subsampling the image (half size)


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
            self.n_samples_per_file_LR = _f['LR_fields'].shape[0]
            self.n_in_channels_LR = _f['LR_fields'].shape[1]
            self.img_shape_x_LR = _f['LR_fields'].shape[2]
            self.img_shape_y_LR = _f['LR_fields'].shape[3]
        self.n_samples_total = self.n_files * self.n_samples_per_file
        self.files = [None for _ in range(self.n_files)]
        self.LR_files = [None for _ in range(self.n_files)]
        print("Number of samples per file: {}".format(self.n_samples_per_file))
        print("Found data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_total, self.img_shape_x, self.img_shape_y, self.n_in_channels))
        print("Found LR data at path {}. Number of examples: {}. Image Shape: {} x {} x {}".format(
            self.location, self.n_samples_per_file_LR, self.img_shape_x_LR, self.img_shape_y_LR, self.n_in_channels_LR))

    def _open_file(self, file_idx):
        _file = h5py.File(self.files_paths[file_idx], 'r')
        self.files[file_idx] = _file['fields']  
        self.LR_files[file_idx] = _file['LR_fields']
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
        X = self.transform(self.LR_files[file_idx][local_idx])
        if self.train:
            # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(y, output_size=(self.crop_size, self.crop_size)) # i,j are the top left corner of the crop, h,w are the height and width of the crop
            y = F.crop(y, i, j, h, w)
            X = F.crop(X, i//self.upscale_factor, j//self.upscale_factor, h//self.upscale_factor, w//self.upscale_factor) # relative location on LR
        return X, y

    def get_indices(self, global_idx):
        if self.train:
            file_idx = int(global_idx/(self.n_samples_per_file*self.n_patches))  # which file we are on
            local_idx = int((global_idx//self.n_patches) % self.n_samples_per_file)  # which sample in that file we are on 
        else:
            file_idx = int(global_idx/self.n_samples_per_file)  # which file we are on
            local_idx = int(global_idx % self.n_samples_per_file)  # which sample in that file we are on 

        return file_idx, local_idx

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
            self.bicubicDown_transform = transforms.Resize((int(self.crop_size/upscale_factor),int(self.crop_size/upscale_factor)),Image.BICUBIC,antialias=False)  # subsampling the image (half size)
        elif (self.train == False) and (method == "bicubic"):
            self.bicubicDown_transform = transforms.Resize((int((self.img_shape_x-1)/upscale_factor),int(self.img_shape_y/upscale_factor)),Image.BICUBIC,antialias=False)  # subsampling the image (half size)

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

    
def GetCosmoSimData(data_path, data_tag, train, transform, crop_size, n_patches):

    if data_tag == '/train':
        hres_data_dir = data_path+'/cosmo_train.h5'
        lres_data_dir = data_path+'/cosmo_train_lres.h5'

    elif data_tag == '/valid_1':
        hres_data_dir = data_path+'/cosmo_val_1.h5'
        lres_data_dir = data_path+'/cosmo_val_1_lres.h5'

    elif data_tag == '/valid_2':
        hres_data_dir = data_path+'/cosmo_val_2.h5'
        lres_data_dir = data_path+'/cosmo_val_2_lres.h5'

    elif data_tag == '/test_1':
        hres_data_dir = data_path+'/cosmo_test_1.h5'
        lres_data_dir = data_path+'/cosmo_test_1_lres.h5'

    elif data_tag == '/test_2':
        hres_data_dir = data_path+'/cosmo_test_2.h5'
        lres_data_dir = data_path+'/cosmo_test_2_lres.h5'
    else:
        raise ValueError('Data tag {} not recognized'.format(data_tag))


    # %---%
    # get hres data info 
    with h5py.File(hres_data_dir, 'r') as _f:
        print("Getting hres data stats from {}".format(hres_data_dir))
        n_samples_hres = _f['fields'].shape[0]
        n_in_channels = _f['fields'].shape[1]
        height_hres = _f['fields'].shape[2]
        width_hres = _f['fields'].shape[3]

    # read hres data
    _file = h5py.File(hres_data_dir, 'r')
    hres = _file['fields'][()]  # [n_smaples,2,4096, 4096]

    # %---%
    # get lres data info 
    with h5py.File(lres_data_dir, 'r') as _f:
        print("Getting lres data stats from {}".format(lres_data_dir))
        n_samples_lres = _f['fields'].shape[0]
        n_in_channels = _f['fields'].shape[1]
        height_lres = _f['fields'].shape[2]
        width_lres = _f['fields'].shape[3]

    _file = h5py.File(lres_data_dir, 'r')
    lres = _file['fields'][()]  # [400,2,512, 512]

    if train == True:
        print('Random cropping...')
        # do random crop and pairs
        lres_dataset, hres_dataset = [], []
        crop_size = int(crop_size/8)
        for i in range(lres.shape[0]):
            for j in range(n_patches):
                left = np.random.randint(1, width_lres - crop_size)
                top = np.random.randint(1, height_lres - crop_size)
                right = left + crop_size
                bottom = top + crop_size

                lres_img = lres[i:(i+1), :, top:bottom, left:right]
                hres_img = hres[i:(i+1), :, (8*top):(8*bottom), (8*left):(8*right)]

                lres_dataset.append(transform(lres_img))
                hres_dataset.append(transform(hres_img))
        
        lres_dataset = torch.cat(lres_dataset, dim=0)
        hres_dataset = torch.cat(hres_dataset, dim=0)
    else:
        # for evaluation
        lres_dataset = transform(lres)
        hres_dataset = transform(hres)

    print('The shape of hres data samples: ', hres_dataset.shape)
    print('The shape of lres data samples: ', lres_dataset.shape)

    return TensorDataset(lres_dataset, hres_dataset)

