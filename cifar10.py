import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import matplotlib.pyplot as plt

from architecture_cls6 import HB_transformer as net
#from perceiver import PerceiverLogits as net
#from perceiver2 import Perceiver as net

train_loader = DataLoader(
                datasets.CIFAR10(
                        "../data/CIFAR10",
                        train=True,
                        download=True,
                        transform=transforms.Compose([
                                transforms.RandomCrop(32, padding=4),
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=128, shuffle=True)#, pin_memory=True)


test_loader = DataLoader(
                datasets.CIFAR10(
                        '../data/CIFAR10',
                        train=False,
                        download=True,
                        transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(
                                        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                                ),
                        ),
                batch_size=64, shuffle=False)#, pin_memory=True)


device = torch.device("cuda:0")
model = net((3,1024), device, output=10).to(device)

gamma = 0.1 ** 0.5
optimizer = optim.Adam(model.parameters(), lr=gamma * 0.01)
#optimizer = optim.SGD(model.parameters(), lr=0.004)
#scheduler = optim.lr_scheduler.StepLR(
#        optimizer, step_size=3, gamma=gamma, last_epoch=-1, verbose=False)

#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [84,102,114], gamma=0.1, last_epoch=-1, verbose=False)

criterion = nn.CrossEntropyLoss()

best_acc = 0
for epoch in range(500):
    runnning_loss = 0
    for x, y in train_loader:
        x = x.reshape(x.shape[0],3,-1).permute(0,2,1) # (B, 1024, 3)
        #x = x.permute(0,2,3,1)
        optimizer.zero_grad()
               
        output, _ = model(x.float().to(device))
        #print((torch.isnan(output)).sum())
        loss = criterion(output, y.long().to(device))
        loss.backward()
        optimizer.step()
        runnning_loss += loss.item()
        #print(loss.item())
    
    #scheduler.step()
        
    runnning_loss /= len(train_loader)
    print("[Epoch:%d] [Loss:%f]" % ((epoch+1), runnning_loss), end=" ")
    
    
    accuracy = 0
    with torch.no_grad():
        model.eval()
        #print("[Scale:%f]" % model.scale, end=" ")
        correct = 0
        for x, y in test_loader:
            x = x.reshape(x.shape[0],3,-1).permute(0,2,1) # (B, 1024, 3)
            #x = x.permute(0,2,3,1)
            output, visual = model(x.float().to(device))
            pred = output.argmax(1, keepdim=True)
            correct += pred.eq(y.long().to(device).view_as(pred)).sum().item()
                
        accuracy = correct / len(test_loader.dataset)

        if accuracy >= best_acc:
            print("[Accuracy:%f] **Best**" % accuracy)
            best_acc = accuracy
        else:
            print("[Accuracy:%f]" % accuracy)
            
        model.train()
        '''
        map_base = visual[0]
        for i in range(32):
            single_map = map_base[:,i].reshape(32,32)
            plt.imshow(single_map.detach().cpu().numpy())
            plt.savefig("/home/compu/ymh/s_modeling/save/cifar/epoch%03d_map%02d.png" % ((epoch+1),i))
        '''
        
    #scheduler.step()
    
    #torch.save({'model_state_dict': model.state_dict()}, "/data/ymh/gpu_test/resnet56_preact.pth")