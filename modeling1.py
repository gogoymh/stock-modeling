import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import os

from Dataset import market5
import config as cfg
from architecture7 import HB_transformer as net
from loss_smp import TverskyLoss

save_path = "/home/compu/ymh/s_modeling/save/"
##############################################################################################################################
input_path = "/home/compu/ymh/s_modeling/data/input_v1/"
output_path = "/home/compu/ymh/s_modeling/data/output_v2/"

dataset = market5(input_path, output_path)

train_sampler = SubsetRandomSampler(cfg.train_idx)
valid_sampler = SubsetRandomSampler(cfg.valid_idx)
n_val_sample = len(cfg.valid_idx)
print(n_val_sample)

train_loader = DataLoader(dataset=dataset, batch_size=128, sampler=train_sampler)
valid_loader = DataLoader(dataset=dataset, batch_size=1, sampler=valid_sampler)


##############################################################################################################################
device = torch.device("cuda:0")
criterion = cfg.DiceLoss()
#criterion = TverskyLoss()

model = net((5,3062), device, repeat=4, share=6, d_models=32, dropout=0)
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.004)

test_idx = 0    

for epoch in range(1000):
    running_loss=0
    model.train()
    for x, y in train_loader:
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

        mean_accuracy = 0
        mean_precision = 0
        mean_recall = 0
        mean_dice = 0

        for x, y in valid_loader:
            x = x.float().to(device)
            y = y >= 0.5
            y = y.to(device)
    
            with torch.no_grad():
                output = model(x)
    
            pred = output >= 0.5        
            pred = pred.view(-1)
                    
            Trues = pred[pred == y.view(-1)]
            Falses = pred[pred != y.view(-1)]
                
            TP = (Trues == 1).sum().item()
            TN = (Trues == 0).sum().item()
            FP = (Falses == 1).sum().item()
            FN = (Falses == 0).sum().item()
        
            accuracy = (TP + TN)/(TP + TN + FP + FN)
            if TP == 0:
                precision = 0
                recall = 0
                dice = 0
            else:
                precision = TP / (TP + FP)
                recall = TP / (TP + FN)
                dice = (2 * TP) / ((2 * TP) + FP + FN)

            mean_accuracy += accuracy
            mean_precision += precision
            mean_recall += recall
            mean_dice += dice
    
        mean_accuracy /= n_val_sample
        mean_precision /= n_val_sample
        mean_recall /= n_val_sample
        mean_dice /= n_val_sample
                
        print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss), end=" ")
        print("[Accuracy:%f]" % mean_accuracy, end=" ")
        print("[Precision:%f]" % mean_precision, end=" ")
        print("[Recall:%f]" % mean_recall, end=" ")
        print("[F1 score:%f]" % mean_dice)
            
        #model_name = os.path.join(save_path, "test_%02d.pth" % test_idx)
        
        #torch.save({'model_state_dict': model.state_dict()}, model_name)