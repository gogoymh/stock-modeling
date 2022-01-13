import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import os
from tqdm import tqdm
import numpy as np

from Dataset import Fiveday_EMA
from architecture6 import HB_transformer as net

save_path = "/home/DATA/ymh/s_modeling/save/marcap2"
##############################################################################################################################
input_path = "/home/DATA/ymh/s_modeling/data/cumulative7"

dataset = Fiveday_EMA(input_path)

total = dataset.len#3668785
n_train_sample = int(total*0.8)#3668785*0.8)

##############################################################################################################################
device = torch.device("cuda:0")
criterion = nn.BCELoss()

for test_idx in range(5):
    print("="*100)
    indices = list(range(total))
    np.random.shuffle(indices)

    train_idx = indices[:n_train_sample]
    valid_idx = indices[n_train_sample:]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    n_val_sample = len(valid_idx)

    train_loader = DataLoader(dataset=dataset, batch_size=8192, sampler=train_sampler, num_workers=32)
    valid_loader = DataLoader(dataset=dataset, batch_size=8192, sampler=valid_sampler, num_workers=32)
        
    #model = net((5,7), device)
    model = net((41,1))
    model.to(device)

    #lr = (8192/256) * 0.0005
    #gamma = 0.1 ** 0.5
    optimizer = optim.Adam(model.parameters(), lr=0.004)#gamma * 0.01)
    
    print("="*100)
    #best_precision = 0
    best_profit = 0
    for epoch in range(80):
        running_loss=0
        model.train()
        for x, y, _, _ in tqdm(train_loader):
            x = x.float().to(device)
            y = y.float().to(device)

            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
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
            print("[Profit:%f]" % mean_profit, end=" ")
            print("[Stop:%f]" % mean_stop)
            
            #if precision >= best_precision:
            if (epoch+1) > 10 and mean_profit >= best_profit:
                if precision != 0 and precision != 1:
                    model_name = os.path.join(save_path, "input_marcap_test_%02d.pth" % test_idx)
                    torch.save({'model_state_dict': model.state_dict()}, model_name)
                    print("model saved.")
                    #best_precision = precision
                    best_profit = mean_profit