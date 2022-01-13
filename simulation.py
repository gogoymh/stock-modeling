import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import os
from tqdm import tqdm
import pandas as pd

from Dataset import Fiveday_EMA
import config as cfg
from architecture6 import HB_transformer as net

save_path = "/home/DATA/ymh/s_modeling/save/marcap2"
len_save = len(os.listdir(save_path))
##############################################################################################################################
input_path = "/home/DATA/ymh/s_modeling/data/cumulative7"

dataset = Fiveday_EMA(input_path)

loader = DataLoader(dataset=dataset, batch_size=8192, shuffle=False, num_workers=32, drop_last=True)

##############################################################################################################################
symbols = pd.read_pickle("symbols.pkl")

device = torch.device("cuda:0")

model = net((41,1))
model.to(device)


mean_real_1 = 0
mean_loss_1 = 0

mean_real_2 = 0
mean_loss_2 = 0

mean_real_3 = 0
mean_loss_3 = 0

mean_real_4 = 0
mean_loss_4 = 0

mean_real_5 = 0
mean_loss_5 = 0

mean_real_6 = 0
mean_loss_6 = 0

mean_real_7 = 0
mean_loss_7 = 0

mean_real_10 = 0
mean_loss_10 = 0

mean_real_15 = 0
mean_loss_15 = 0

mean_real_20 = 0
mean_loss_20 = 0

print("="*100)
print("="*100)
print("="*100)
for test_idx in range(len_save):
    print("="*40, end=" ")
    print("[Test:%d]" % test_idx, end=" ")
    print("="*40)
    model_name = os.path.join(save_path, "input_marcap_test_%02d.pth" % test_idx)
    checkpoint = torch.load(model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    total = 0
    
    real_1 = 0
    loss_1 = 0
    
    real_2 = 0
    loss_2 = 0
    
    real_3 = 0
    loss_3 = 0
    
    real_4 = 0
    loss_4 = 0
    
    real_5 = 0
    loss_5 = 0
    
    real_6 = 0
    loss_6 = 0
    
    real_7 = 0
    loss_7 = 0
    
    real_10 = 0
    loss_10 = 0
    
    real_15 = 0
    loss_15 = 0
    
    real_20 = 0
    loss_20 = 0
    
    for x, y, profit, stop in loader:
        x = x.float().to(device)
        y = y.float().to(device)
        profit = profit.float().to(device)
        stop = stop.float().to(device)
    
        with torch.no_grad():
            output = model(x)
    
        pred = output >= 0.5
        num = pred.sum().item()
        
        get = profit[pred]
        loss = stop[pred]
        
        total += num
        
        real_1 += (get >= 1).sum().item()
        loss_1 += loss[get < 1].sum().item()
        
        real_2 += (get >= 2).sum().item()
        loss_2 += loss[get < 2].sum().item()
        
        real_3 += (get >= 3).sum().item()
        loss_3 += loss[get < 3].sum().item()
        
        real_4 += (get >= 4).sum().item()
        loss_4 += loss[get < 4].sum().item()
        
        real_5 += (get >= 5).sum().item()
        loss_5 += loss[get < 5].sum().item()
        
        real_6 += (get >= 6).sum().item()
        loss_6 += loss[get < 6].sum().item()
        
        real_7 += (get >= 7).sum().item()
        loss_7 += loss[get < 7].sum().item()
        
        real_10 += (get >= 10).sum().item()
        loss_10 += loss[get < 10].sum().item()
        
        real_15 += (get >= 15).sum().item()
        loss_15 += loss[get < 15].sum().item()
        
        real_20 += (get >= 20).sum().item()
        loss_20 += loss[get < 20].sum().item()
        
        
    real_1 /= total
    loss_1 /= total
    
    real_2 /= total
    loss_2 /= total
    
    real_3 /= total
    loss_3 /= total
    
    real_4 /= total
    loss_4 /= total
    
    real_5 /= total
    loss_5 /= total
    
    real_6 /= total
    loss_6 /= total
    
    real_7 /= total
    loss_7 /= total
    
    real_10 /= total
    loss_10 /= total
    
    real_15 /= total
    loss_15 /= total
    
    real_20 /= total
    loss_20 /= total
        
    mean_real_1 += real_1
    mean_loss_1 += loss_1
    
    mean_real_2 += real_2
    mean_loss_2 += loss_2
    
    mean_real_3 += real_3
    mean_loss_3 += loss_3
    
    mean_real_4 += real_4
    mean_loss_4 += loss_4
    
    mean_real_5 += real_5
    mean_loss_5 += loss_5
    
    mean_real_6 += real_6
    mean_loss_6 += loss_6
    
    mean_real_7 += real_7
    mean_loss_7 += loss_7
    
    mean_real_10 += real_10
    mean_loss_10 += loss_10
    
    mean_real_15 += real_15
    mean_loss_15 += loss_15
    
    mean_real_20 += real_20
    mean_loss_20 += loss_20
    

mean_real_1 /= len_save
mean_loss_1 /= len_save

mean_real_2 /= len_save
mean_loss_2 /= len_save

mean_real_3 /= len_save 
mean_loss_3 /= len_save

mean_real_4 /= len_save
mean_loss_4 /= len_save

mean_real_5 /= len_save
mean_loss_5 /= len_save

mean_real_6 /= len_save
mean_loss_6 /= len_save

mean_real_7 /= len_save
mean_loss_7 /= len_save

mean_real_10 /= len_save
mean_loss_10 /= len_save

mean_real_15 /= len_save
mean_loss_15 /= len_save

mean_real_20 /= len_save
mean_loss_20 /= len_save

print("="*100)
print(1*mean_real_1 + mean_loss_1*(1-mean_real_1))
print(2*mean_real_2 + mean_loss_2*(1-mean_real_2))
print(3*mean_real_3 + mean_loss_3*(1-mean_real_3))
print(4*mean_real_4 + mean_loss_4*(1-mean_real_4))
print(5*mean_real_5 + mean_loss_5*(1-mean_real_5))
print(6*mean_real_6 + mean_loss_6*(1-mean_real_6))
print(7*mean_real_7 + mean_loss_7*(1-mean_real_7))
print(10*mean_real_10 + mean_loss_10*(1-mean_real_10))
print(15*mean_real_15 + mean_loss_15*(1-mean_real_15))
print(20*mean_real_20 + mean_loss_20*(1-mean_real_20))









