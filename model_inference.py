import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import os
from tqdm import tqdm
import pandas as pd

from Dataset import inference_set
import config as cfg
from architecture6 import HB_transformer as net

save_path = "/home/DATA/ymh/s_modeling/save/marcap2"
len_save = len(os.listdir(save_path))
##############################################################################################################################
input_path = "/home/DATA/ymh/s_modeling/data/marcap_20210614"
filename = os.listdir(input_path)
filename.sort()

dataset = inference_set(input_path)

test_loader = DataLoader(dataset=dataset, batch_size=2434, shuffle=False)

##############################################################################################################################
symbols = pd.read_csv("/home/DATA/ymh/s_modeling/data/data_2822_20210608.csv", encoding='utf-8')
symbols = symbols.rename(columns={"단축코드":"Code", "시장구분":"Market"})
symbols = pd.concat([symbols['Code'], symbols['Market']], axis=1)
symbols = symbols[symbols.Market != 'KONEX']

device = torch.device("cuda:0")

model = net((41,1))
model.to(device)

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

    for x in tqdm(test_loader):
        x = x.float().to(device)
    
        with torch.no_grad():
            output = model(x)
    

    pred = torch.topk(output, 10)[0]
    pred = pred[pred >= 0.55].tolist()
    index = torch.topk(output, 10)[1]
    index = index[:len(pred)].tolist()
    
    #index = (pred == 1).nonzero(as_tuple=True)[0].tolist()

    print(index)
    print(pred)

    for i in index:
        print(symbols.iloc[i,0], filename[i])


