import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

#******************************************************************************
# Read in data
#******************************************************************************
def getData(name, train_bs=128, test_bs=256):

    if name == 'doublegyre4':
       D = torch.load('../datasets/doublegyre/doublegyre_train.npy') 
       train_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('../datasets/doublegyre/doublegyre_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/doublegyre/doublegyre_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/doublegyre/doublegyre_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/doublegyre/doublegyre_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader   

    if name == 'doublegyre8':
       D = torch.load('../datasets/doublegyre/doublegyre_train.npy') 
       train_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('../datasets/doublegyre/doublegyre_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/doublegyre/doublegyre_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/doublegyre/doublegyre_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/doublegyre/doublegyre_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader     

    if name == 'isoflow4':
       D = torch.load('../datasets/isoflow/isoflow_train.npy') 
       train_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('../datasets/isoflow/isoflow_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/isoflow/isoflow_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/isoflow/isoflow_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/isoflow/isoflow_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader     
 

    if name == 'isoflow8':
       D = torch.load('../datasets/isoflow/isoflow_train.npy') 
       train_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('../datasets/isoflow/isoflow_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/isoflow/isoflow_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/isoflow/isoflow_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/isoflow/isoflow_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader     
 

    if name == 'rbc4':
       D = torch.load('../datasets/rbc/rbc_train.npy') 
       train_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('../datasets/rbc/rbc_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/rbc/rbc_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/rbc/rbc_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/rbc/rbc_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[1], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader   

    if name == 'rbc8':
       D = torch.load('../datasets/rbc/rbc_train.npy') 
       train_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=True)
       D = torch.load('../datasets/rbc/rbc_test_1.npy') 
       test1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/rbc/rbc_val_1.npy') 
       val1_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/rbc/rbc_test_2.npy') 
       test2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       D = torch.load('../datasets/rbc/rbc_val_2.npy') 
       val2_loader = DataLoader(TensorDataset(D[0], D[2]), batch_size=train_bs, shuffle=False)
       return train_loader, test1_loader, val1_loader, test2_loader, val2_loader     
 
                

    else:
        raise ValueError('dataset {} not recognized'.format(name))






    
    