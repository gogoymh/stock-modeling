import numpy as np
import os
import torch
import torch.nn as nn

##############################################################################################################################
'''
indices = list(range(5820))
np.random.shuffle(indices)

train_idx = indices[:4656]
valid_idx = indices[4656:]

np.save(os.path.join("C:\\유민형\\개인 연구\\stock_modeling", "train_idx"), train_idx)
np.save(os.path.join("C:\\유민형\\개인 연구\\stock_modeling", "valid_idx"), valid_idx)
'''

'''
#path = "C:\\유민형\\개인 연구\\stock_modeling"
path = "/home/compu/ymh/s_modeling"
train_idx = np.load(os.path.join(path,'train_idx.npy')).tolist()
valid_idx = np.load(os.path.join(path,'valid_idx.npy')).tolist()


#path = "/home/compu/ymh/s_modeling"
train6_idx = np.load(os.path.join(path,'train6_idx.npy')).tolist()
valid6_idx = np.load(os.path.join(path,'valid6_idx.npy')).tolist()


train11_idx = np.load(os.path.join(path,'train11_idx.npy')).tolist()
valid11_idx = np.load(os.path.join(path,'valid11_idx.npy')).tolist()
'''
'''
path = "/data/ymh/s_modeling"
train_3_2_idx = np.load(os.path.join(path,'train_3_2_idx.npy')).tolist()
valid_3_2_idx = np.load(os.path.join(path,'valid_3_2_idx.npy')).tolist()

train5_3_2_idx = np.load(os.path.join(path,'train5_3_2_idx.npy')).tolist()
valid5_3_2_idx = np.load(os.path.join(path,'valid5_3_2_idx.npy')).tolist()

train_s3_idx = np.load(os.path.join(path,'train_s3_idx.npy')).tolist()
valid_s3_idx = np.load(os.path.join(path,'valid_s3_idx.npy')).tolist()

train5_s3_idx = np.load(os.path.join(path,'train5_s3_idx.npy')).tolist()
valid5_s3_idx = np.load(os.path.join(path,'valid5_s3_idx.npy')).tolist()
'''
#path = "/home/DATA/ymh/s_modeling"
#train10_s3_idx = np.load(os.path.join(path,'train10_s3_idx.npy')).tolist()
#valid10_s3_idx = np.load(os.path.join(path,'valid10_s3_idx.npy')).tolist()

#path = "/data/ymh/s_modeling"
#train_s3_idx = np.load(os.path.join(path,'train_s3_idx.npy')).tolist()
#valid_s3_idx = np.load(os.path.join(path,'valid_s3_idx.npy')).tolist()

path = "/home/DATA/ymh/s_modeling"
train_s3_idx = np.load(os.path.join(path,'train_s3_idx.npy')).tolist()
valid_s3_idx = np.load(os.path.join(path,'valid_s3_idx.npy')).tolist()

train_ema_idx = np.load(os.path.join(path,'train_ema_idx.npy')).tolist()
valid_ema_idx = np.load(os.path.join(path,'valid_ema_idx.npy')).tolist()

train_ema5_idx = np.load(os.path.join(path,'train_ema5_idx.npy')).tolist()
valid_ema5_idx = np.load(os.path.join(path,'valid_ema5_idx.npy')).tolist()
##############################################################################################################################
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        
        smooth = 0
        # m1=pred.flatten()
        # m2=target.flatten()
        # intersection = (m1 * m2)

        # score=1-((2. * torch.sum(intersection) + smooth) / (torch.sum(m1) + torch.sum(m2) + smooth))
        # #score=1-((2. * torch.sum(intersection) + smooth) / (torch.sum(m1*m1) + torch.sum(m2*m2) + smooth))
                
        num = target.shape[0]
        m1 = pred.view(num, -1)
        m2 = target.view(num, -1)
        intersection=torch.mul(m1,m2)
        score = 1-torch.sum((2. * torch.sum(intersection,dim=1) + smooth) / (torch.sum(m1,dim=1) + torch.sum(m2,dim=1) + smooth))/num
        
        # for squared
        ## score = 1-torch.sum((2. * torch.sum(intersection,dim=1) + smooth) / (torch.sum(m1*m1,dim=1) + torch.sum(m2*m2,dim=1) + smooth))/num
        
        return score
    
if __name__ == "__main__":
    a = torch.rand((3,3062))
    a.requires_grad = True
    b = torch.rand((3,3062))
    
    c = DiceLoss()
    
    d = c(a,b)
    print(d)