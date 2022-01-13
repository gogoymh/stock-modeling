import torch
from torch.utils.data import Dataset
import os
import numpy as np

class inference_set(Dataset):
    def __init__(self, path1):
        super().__init__()
        
        self.path1 = path1
        self.names = os.listdir(self.path1)
        self.names.sort()
        
        self.len = len(self.names)
        
        print("Length: " + str(self.len))
        
    def __getitem__(self, index):
        input_array = np.load(os.path.join(self.path1, self.names[index]))
        input_array = torch.from_numpy(input_array)
        
        return input_array
    
    def __len__(self):
        return self.len

class Fiveday_EMA(Dataset):
    def __init__(self, path1):
        super().__init__()
        
        self.path1 = path1
        self.names = os.listdir(self.path1)
        self.names.sort()
        
        self.len = len(self.names)
        
        print("Length: " + str(self.len))
        
    def __getitem__(self, index):
        input_array = np.load(os.path.join(self.path1, self.names[index]))
        input_array = torch.from_numpy(input_array)
        output = float(self.names[index].split('_')[2])
        
        profit = float(self.names[index].split('_')[3])
        stop = float(self.names[index].split('_')[4])
        
        return input_array, output, profit, stop
    
    def __len__(self):
        return self.len


class Threeday_EMA(Dataset):
    def __init__(self, path1):
        super().__init__()
        
        self.path1 = path1
        self.names = os.listdir(self.path1)
        self.names.sort()
        
        self.len = len(self.names)
        
        print("Length: " + str(self.len))
        
    def __getitem__(self, index):
        input_array = np.load(os.path.join(self.path1, self.names[index]))
        input_array = torch.from_numpy(input_array)
        output = float(self.names[index].split('_')[2])
        
        profit = float(self.names[index].split('_')[3])
        
        return input_array, output, profit
    
    def __len__(self):
        return self.len

class Threeday10_3(Dataset):
    def __init__(self, path1):
        super().__init__()
        
        self.path1 = path1
        self.names = os.listdir(self.path1)
        self.names.sort()
        
        self.len = len(self.names) - 10
        
        print("Length: " + str(self.len))
        
    def __getitem__(self, index):
        input_1 = np.load(os.path.join(self.path1, self.names[index]))
        input_2 = np.load(os.path.join(self.path1, self.names[index+1]))
        input_3 = np.load(os.path.join(self.path1, self.names[index+2]))
        input_4 = np.load(os.path.join(self.path1, self.names[index+3]))
        input_5 = np.load(os.path.join(self.path1, self.names[index+4]))
        input_6 = np.load(os.path.join(self.path1, self.names[index+5]))
        input_7 = np.load(os.path.join(self.path1, self.names[index+6]))
        input_8 = np.load(os.path.join(self.path1, self.names[index+7]))
        input_9 = np.load(os.path.join(self.path1, self.names[index+8]))
        input_10 = np.load(os.path.join(self.path1, self.names[index+9]))
        
        input_array = np.concatenate((input_1,input_2,input_3,input_4,input_5,
                                      input_6,input_7,input_8,input_9,input_10), axis=1)
        input_array = torch.from_numpy(input_array)
        output = float(self.names[index+9].split('_')[2])
        
        return input_array, output
    
    def __len__(self):
        return self.len

class Threeday5_3(Dataset):
    def __init__(self, path1):
        super().__init__()
        
        self.path1 = path1
        self.names = os.listdir(self.path1)
        self.names.sort()
        
        self.len = len(self.names) - 5
        
        print("Length: " + str(self.len))
        
    def __getitem__(self, index):
        input_1 = np.load(os.path.join(self.path1, self.names[index]))
        input_2 = np.load(os.path.join(self.path1, self.names[index+1]))
        input_3 = np.load(os.path.join(self.path1, self.names[index+2]))
        input_4 = np.load(os.path.join(self.path1, self.names[index+3]))
        input_5 = np.load(os.path.join(self.path1, self.names[index+4]))
        
        input_array = np.concatenate((input_1,input_2,input_3,input_4,input_5), axis=1)
        input_array = torch.from_numpy(input_array)
        output = float(self.names[index+4].split('_')[2])
        
        return input_array, output
    
    def __len__(self):
        return self.len

class Threeday_3(Dataset):
    def __init__(self, path1):
        super().__init__()
        
        self.path1 = path1
        self.names = os.listdir(self.path1)
        self.names.sort()
        
        self.len = len(self.names)
        
        print("Length: " + str(self.len))
        
    def __getitem__(self, index):
        input_array = np.load(os.path.join(self.path1, self.names[index]))
        input_array = torch.from_numpy(input_array)
        output = float(self.names[index].split('_')[2])
        #output = torch.FloatTensor([output])
        
        return input_array, output
    
    def __len__(self):
        return self.len

class Oneday5_3_2(Dataset):
    def __init__(self, path1):
        super().__init__()
        
        self.path1 = path1
        self.names = os.listdir(self.path1)
        self.names.sort()
        
        self.len = len(self.names) - 5
        
        print("Length: " + str(self.len))
        
    def __getitem__(self, index):
        input_1 = np.load(os.path.join(self.path1, self.names[index]))
        input_2 = np.load(os.path.join(self.path1, self.names[index+1]))
        input_3 = np.load(os.path.join(self.path1, self.names[index+2]))
        input_4 = np.load(os.path.join(self.path1, self.names[index+3]))
        input_5 = np.load(os.path.join(self.path1, self.names[index+4]))
        
        input_array = np.concatenate((input_1,input_2,input_3,input_4,input_5), axis=1)
        input_array = torch.from_numpy(input_array)
        output = float(self.names[index+4].split('.')[0].split('_')[2])
        
        return input_array, output
    
    def __len__(self):
        return self.len

class Oneday_3_2(Dataset):
    def __init__(self, path1):
        super().__init__()
        
        self.path1 = path1
        self.names = os.listdir(self.path1)
        self.names.sort()
        
        self.len = len(self.names)
        
        print("Length: " + str(self.len))
        
    def __getitem__(self, index):
        input_array = np.load(os.path.join(self.path1, self.names[index]))
        input_array = torch.from_numpy(input_array)
        output = float(self.names[index].split('.')[0].split('_')[2])
        #output = torch.FloatTensor([output])
        
        return input_array, output
    
    def __len__(self):
        return self.len

class market5_11(Dataset):
    def __init__(self, path1, path2):
        super().__init__()
        
        self.path1 = path1
        self.path2 = path2
        
        self.len = 529
        
        print("Length: " + str(self.len))
        
    def __getitem__(self, index):
        
        input_1 = np.load(os.path.join(self.path1,'input_%05d.npy' % (11*index)))
        input_2 = np.load(os.path.join(self.path1,'input_%05d.npy' % (11*index+1)))
        input_3 = np.load(os.path.join(self.path1,'input_%05d.npy' % (11*index+2)))
        input_4 = np.load(os.path.join(self.path1,'input_%05d.npy' % (11*index+3)))
        input_5 = np.load(os.path.join(self.path1,'input_%05d.npy' % (11*index+4)))
        input_6 = np.load(os.path.join(self.path1,'input_%05d.npy' % (11*index+5)))
        input_7 = np.load(os.path.join(self.path1,'input_%05d.npy' % (11*index+6)))
        input_8 = np.load(os.path.join(self.path1,'input_%05d.npy' % (11*index+7)))
        input_9 = np.load(os.path.join(self.path1,'input_%05d.npy' % (11*index+8)))
        input_10 = np.load(os.path.join(self.path1,'input_%05d.npy' % (11*index+9)))
        
        input_array = np.concatenate((input_1,input_2,input_3,input_4,input_5,
                                      input_6,input_7,input_8,input_9,input_10), axis=1)
        output_array = np.load(os.path.join(self.path2,'output_%05d.npy' % (11*index+9)))

        input_array = torch.from_numpy(input_array)
        output_array = torch.from_numpy(output_array)
        
        return input_array, output_array
    
    def __len__(self):
        return self.len

class market5_6(Dataset):
    def __init__(self, path1, path2):
        super().__init__()
        
        self.path1 = path1
        self.path2 = path2
        
        self.len = 970
        
        print("Length: " + str(self.len))
        
    def __getitem__(self, index):
        
        input_1 = np.load(os.path.join(self.path1,'input_%05d.npy' % (6*index)))
        input_2 = np.load(os.path.join(self.path1,'input_%05d.npy' % (6*index+1)))
        input_3 = np.load(os.path.join(self.path1,'input_%05d.npy' % (6*index+2)))
        input_4 = np.load(os.path.join(self.path1,'input_%05d.npy' % (6*index+3)))
        input_5 = np.load(os.path.join(self.path1,'input_%05d.npy' % (6*index+4)))
        
        input_array = np.concatenate((input_1,input_2,input_3,input_4,input_5), axis=1)
        output_array = np.load(os.path.join(self.path2,'output_%05d.npy' % (6*index+4)))

        input_array = torch.from_numpy(input_array)
        output_array = torch.from_numpy(output_array)
        
        return input_array, output_array
    
    def __len__(self):
        return self.len

class market5(Dataset):
    def __init__(self, path1, path2):
        super().__init__()
        
        self.path1 = path1
        self.path2 = path2
        
        self.len = len(os.listdir(self.path2))
        
        print("Length: " + str(self.len))
        
    def __getitem__(self, index):
        
        input_array = np.load(os.path.join(self.path1,'input_%05d.npy' % index))
        output_array = np.load(os.path.join(self.path2,'output_%05d.npy' % index))

        input_array = torch.from_numpy(input_array)
        output_array = torch.from_numpy(output_array)
        
        return input_array, output_array
    
    def __len__(self):
        return self.len
    
if __name__ == "__main__":   
    '''
    path1 = "/home/compu/ymh/s_modeling/data/input_v1/"
    path2 = "/home/compu/ymh/s_modeling/data/output_v3/"
    
    a = market5(path1, path2)
    
    '''
    #path = "/data/ymh/s_modeling/data/input_v4"
    path = "/home/DATA/ymh/s_modeling/data/input_v12"
    
    a = Fiveday_EMA(path)
    
    index = 0
    b, c, d, e = a.__getitem__(index)
    
    print(b.shape)
    print(b)
    print(c)
    print(d)
    print(e)
    '''
    
    _, c = a.__getitem__(0)
    cnt = 0
    min_value = c.sum()
    max_value = c.sum()
    for index in range(a.len):
        _, c = a.__getitem__(index)
        cnt += c.sum()
        
        if c.sum() < min_value:
            min_value = c.sum()
            
        if c.sum() > max_value:
            max_value = c.sum()
    
    print(cnt/a.len)
    print(min_value, max_value)
    '''