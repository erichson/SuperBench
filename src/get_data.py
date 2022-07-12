import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

#******************************************************************************
# Read in data
#******************************************************************************
def getData(name, train_bs=128, test_bs=256):
    if name == 'isoflow':
       train_loader = DataLoader(torch.load('datasets/isoflow/isoflow_train_s4.npy'), batch_size=train_bs, shuffle=True)
       test1_loader = DataLoader(torch.load('datasets/isoflow/isoflow_test_1_s4.npy'), batch_size=test_bs, shuffle=False)
       val1_loader = DataLoader(torch.load('datasets/isoflow/isoflow_val_1_s4.npy'), batch_size=test_bs, shuffle=False)
       test2_loader = DataLoader(torch.load('datasets/isoflow/isoflow_test_2_s4.npy'), batch_size=test_bs, shuffle=False)
       val2_loader = DataLoader(torch.load('datasets/isoflow/isoflow_val_2_s4.npy'), batch_size=test_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader

    if name == 'DoubleGyre':
       train_loader = DataLoader(torch.load('datasets/DoubleGyre/DoubleGyre_train_s4.npy'), batch_size=train_bs, shuffle=True)
       test1_loader = DataLoader(torch.load('datasets/DoubleGyre/DoubleGyre_test_1_s4.npy'), batch_size=test_bs, shuffle=False)
       val1_loader = DataLoader(torch.load('datasets/DoubleGyre/DoubleGyre_val_1_s4.npy'), batch_size=test_bs, shuffle=False)
       test2_loader = DataLoader(torch.load('datasets/DoubleGyre/DoubleGyre_test_2_s4.npy'), batch_size=test_bs, shuffle=False)
       val2_loader = DataLoader(torch.load('datasets/DoubleGyre/DoubleGyre_val_2_s4.npy'), batch_size=test_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader    

    if name == 'RBC':
       train_loader = DataLoader(torch.load('datasets/RBC/RBC_train_s8.npy'), batch_size=train_bs, shuffle=True)
       test1_loader = DataLoader(torch.load('datasets/RBC/RBC_test_1_s8.npy'), batch_size=test_bs, shuffle=False)
       val1_loader = DataLoader(torch.load('datasets/RBC/RBC_val_1_s8.npy'), batch_size=test_bs, shuffle=False)
       test2_loader = DataLoader(torch.load('datasets/RBC/RBC_test_2_s8.npy'), batch_size=test_bs, shuffle=False)
       val2_loader = DataLoader(torch.load('datasets/RBC/RBC_val_2_s8.npy'), batch_size=test_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader    
        

    else:
        raise ValueError('dataset {} not recognized'.format(name))






    
    