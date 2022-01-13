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

#save_path = "/home/compu/ymh/s_modeling/save/"
##############################################################################################################################
#input_path = "/home/compu/ymh/s_modeling/data/input_v1/"

#input_path = "/data/ymh/s_modeling/data/input_v4"
input_path = "/home/DATA/ymh/s_modeling/data/input_v14"

dataset = Fiveday_EMA(input_path)

train_sampler = SubsetRandomSampler(cfg.train_ema5_idx)
valid_sampler = SubsetRandomSampler(cfg.valid_ema5_idx)
n_val_sample = len(cfg.valid_ema5_idx)
print(n_val_sample)

train_loader = DataLoader(dataset=dataset, batch_size=8192, sampler=train_sampler, num_workers=32)
valid_loader = DataLoader(dataset=dataset, batch_size=8192, sampler=valid_sampler, num_workers=32)


##############################################################################################################################
device = torch.device("cuda:0")
#criterion1 = nn.BCELoss()
#criterion2 = cfg.DiceLoss()
criterion = nn.CrossEntropyLoss()

model = net((7,5), output=9)
model.to(device)

#lr = (8192/256) * 0.0005
lr = 0.004
optimizer = optim.Adam(model.parameters(), lr=lr)

#test_idx = 0    

best_acc = 0
for epoch in range(300):
    running_loss=0
    model.train()
    for x, y, _, _ in tqdm(train_loader):
        x = x.float().to(device)
        y = y.long().to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        #print(loss.item())
            
    running_loss /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), running_loss), end=" ")
    
    accuracy = 0
    with torch.no_grad():
        model.eval()
        #print("[Scale:%f]" % model.scale, end=" ")
        correct = 0
        for x, y, _, _ in valid_loader:
            output = model(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
        accuracy = correct / len(valid_loader.dataset)

        if accuracy >= best_acc:
            print("[Accuracy:%f] **Best**" % accuracy)
            best_acc = accuracy
        else:
            print("[Accuracy:%f]" % accuracy)
        model.train()
            
        #model_name = os.path.join(save_path, "test_%02d.pth" % test_idx)
        
        #torch.save({'model_state_dict': model.state_dict()}, model_name)