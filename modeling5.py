import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import os
from tqdm import tqdm

from Dataset import Fiveday_EMA
import config as cfg
from architecture6 import HB_transformer as net

save_path = "/home/DATA/ymh/s_modeling/save/"
##############################################################################################################################
#input_path = "/home/compu/ymh/s_modeling/data/input_v1/"

#input_path = "/data/ymh/s_modeling/data/input_v4"
input_path = "/home/DATA/ymh/s_modeling/data/input_v13"

dataset = Fiveday_EMA(input_path)

train_sampler = SubsetRandomSampler(cfg.train_ema5_idx)
valid_sampler = SubsetRandomSampler(cfg.valid_ema5_idx)
n_val_sample = len(cfg.valid_ema5_idx)
print(n_val_sample)

train_loader = DataLoader(dataset=dataset, batch_size=8192, sampler=train_sampler, num_workers=32)
valid_loader = DataLoader(dataset=dataset, batch_size=8192, sampler=valid_sampler, num_workers=32)


##############################################################################################################################
device = torch.device("cuda:0")
criterion1 = nn.BCELoss()
#criterion2 = cfg.DiceLoss()

#model = net((5,7), device)
model = net((7,5))
model.to(device)

lr = (8192/256) * 0.0005
#gamma = 0.1 ** 0.5
optimizer = optim.Adam(model.parameters(), lr=lr)# lr=0.004)#gamma * 0.01)

test_idx = 0

for epoch in range(90):
    running_loss=0
    model.train()
    for x, y, _, _ in tqdm(train_loader):
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
        mean_stop = 0

        for x, y, profit, stop in tqdm(valid_loader):
            x = x.float().to(device)
            y = y.float().to(device)
            profit = profit.float().to(device)
            stop = stop.float().to(device)
    
            with torch.no_grad():
                output = model(x)
    
            pred = output >= 0.6
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
            
            
            mean_profit += profit[pred].sum().item()
            mean_stop += stop[pred].sum().item()
        
        positive = cum_TP + cum_FP
        if positive != 0:
            mean_profit = mean_profit/positive
            mean_stop = mean_stop/positive
        
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
        print("[Stop:%f]" % mean_stop)
        #print("[F1 score:%f]" % dice)
            
        model_name = os.path.join(save_path, "input_v13_test_%02d.pth" % test_idx)
        
        #torch.save({'model_state_dict': model.state_dict()}, model_name)