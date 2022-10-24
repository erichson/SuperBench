import glob
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
import h5py

def get_data_loader(path, batch_size, train):
  transform = torch.from_numpy
  dataset = GetDataset(path, train, transform) 
  dataloader = DataLoader(dataset,
                          batch_size = int(batch_size),
                          num_workers = 6, # make a param
                          shuffle = (train == True),
                          sampler = None,
                          drop_last = False,
                          pin_memory = torch.cuda.is_available())

  # also return dataset if need more infor from here

  return dataloader

class GetDataset(Dataset):
  def __init__(self, location, train, transform):
    self.location = location
    self.train = train
    self.transform = transform
    self._get_files_stats()

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
    return self.n_samples_total

  def __getitem__(self, global_idx):
    year_idx = int(global_idx/self.n_samples_per_year)  # which year we are on
    local_idx = int(global_idx%self.n_samples_per_year) # which sample in that year we are on 

    #open image file
    if self.files[year_idx] is None:
        self._open_file(year_idx)

    X = self.transform(self.files[year_idx][local_idx])
    y = self.transform(self.files[year_idx][local_idx]) # snafu: change to downsample

    return X, y

