import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import os
from tqdm import tqdm

from Dataset import Threeday_EMA
import config as cfg
from architecture6 import HB_transformer as net

#save_path = "/home/compu/ymh/s_modeling/save/"
##############################################################################################################################
#input_path = "/home/compu/ymh/s_modeling/data/input_v1/"

#input_path = "/data/ymh/s_modeling/data/input_v4"
input_path = "/home/DATA/ymh/s_modeling/data/input_v5"

dataset = Threeday_EMA(input_path)

train_sampler = SubsetRandomSampler(cfg.train_ema_idx)
valid_sampler = SubsetRandomSampler(cfg.valid_ema_idx)
n_val_sample = len(cfg.valid_ema_idx)
print(n_val_sample)

train_loader = DataLoader(dataset=dataset, batch_size=8192, sampler=train_sampler, num_workers=32)
valid_loader = DataLoader(dataset=dataset, batch_size=8192, sampler=valid_sampler, num_workers=32)


##############################################################################################################################
device = torch.device("cuda:0")
criterion1 = nn.BCELoss()
#criterion2 = cfg.DiceLoss()

model = net((6,5))
model.to(device)

lr = (8192/256) * 0.0005
#lr = 0.004
optimizer = optim.AdamW(model.parameters(), lr=lr)

#test_idx = 0    

for epoch in range(300):
    running_loss=0
    model.train()
    for x, y, _ in tqdm(train_loader):
        x = x.float().to(device)
        y = y.float().to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion1(output, y)# criterion1(output, y) + 
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        #print(loss.item())
            
    running_loss /= len(train_loader)
    
    if (epoch+1) % 1 == 0:
        model.eval()

        cum_TP = 0
        cum_TN = 0
        cum_FP = 0
        cum_FN = 0

        mean_profit = 0
        fail_rate = 0

        for x, y, z in tqdm(valid_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            z = z.float().to(device)
    
            with torch.no_grad():
                output = model(x)
    
            pred = output >= 0.5
            #pred = pred.view(-1)
                    
            Trues = pred[pred == y]#.view(-1)]
            Falses = pred[pred != y]#.view(-1)]
                
            TP = (Trues == 1).sum().item()
            TN = (Trues == 0).sum().item()
            FP = (Falses == 1).sum().item()
            FN = (Falses == 0).sum().item()
        
            cum_TP += TP
            cum_TN += TN
            cum_FP += FP
            cum_FN += FN
            
            Falses_z = z[pred != y]
            FP_z = Falses_z[Falses == 1]
            
            mean_profit += FP_z.sum().item()
            fail_rate += (FP_z == 0).sum().item()
        
        if cum_FP != 0:
            mean_profit = mean_profit/cum_FP
            fail_rate = fail_rate/cum_FP
        
        accuracy = (cum_TP + cum_TN)/(cum_TP + cum_TN + cum_FP + cum_FN)
        if cum_TP == 0:
            precision = 0
            recall = 0
            dice = 0
        else:
            precision = cum_TP / (cum_TP + cum_FP)
            recall = cum_TP / (cum_TP + cum_FN)
            dice = (2 * cum_TP) / ((2 * cum_TP) + cum_FP + cum_FN)
                
        print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss), end=" ")
        print("[Accuracy:%f]" % accuracy, end=" ")
        print("[Precision:%f]" % precision, end=" ")
        #print("[Recall:%f]" % recall, end=" ")
        print("[Profit:%f]" % mean_profit, end=" ")
        print("[Fail:%f]" % fail_rate)
        #print("[F1 score:%f]" % dice)
            
        #model_name = os.path.join(save_path, "test_%02d.pth" % test_idx)
        
        #torch.save({'model_state_dict': model.state_dict()}, model_name)