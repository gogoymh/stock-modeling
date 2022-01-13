from marcap import marcap_data
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class Replay_Memory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.zeros((self.capacity,5))
        
        self.buffer_idx = 0
        self.n_entries = 0
        
    def add(self, Open, High, Low, Close, Volume):
        self.buffer[self.buffer_idx, 0] = Open
        self.buffer[self.buffer_idx, 1] = High
        self.buffer[self.buffer_idx, 2] = Low
        self.buffer[self.buffer_idx, 3] = Close
        self.buffer[self.buffer_idx, 4] = Volume
        
        self.buffer_idx += 1
        
        if self.buffer_idx >= self.capacity: # First in, First out
            self.buffer_idx = 0            
        
        if self.n_entries < self.capacity: # How many transitions are stored in buffer
            self.n_entries += 1
            
    def get_mean(self):
        return self.buffer[0:self.n_entries,:].mean(axis=0)

symbols = pd.read_csv("/home/DATA/ymh/s_modeling/data/data_2822_20210608.csv", encoding='utf-8')
symbols = symbols.rename(columns={"단축코드":"Code", "시장구분":"Market"})
symbols = pd.concat([symbols['Code'], symbols['Market']], axis=1)
symbols = symbols[symbols.Market != 'KONEX']

len_stocks = len(symbols) # 종목의 총 개수

df = marcap_data('2015-06-15', '2021-06-14')
df = df[df.Market != 'KONEX']
df = df.drop(['Name', 'Market', 'MarketId', 'Rank',
              'Dept', 'Marcap', 'ChangeCode', 'Changes',
              'ChagesRatio', 'Amount'], axis=1)
df['Date'] = df.index
df['DayOfWeek'] = df['Date'].dt.day_name()
df = df.drop(['Date'], axis=1)
df = pd.get_dummies(df, columns = ['DayOfWeek'])



'''
df_tmp = df[symbols.iloc[0,0] == df['Code']]
save_path = os.path.join("/home/DATA/ymh/s_modeling/data/cumulative",symbols.iloc[0,0])
if os.path.isdir(save_path):
    print("Save path exists: ",save_path)
else:
    os.makedirs(save_path)
    print("Save path is created: ",save_path)
day_path = os.path.join(save_path, str(df.index.values[0])[:10])
if os.path.isdir(day_path):
    print("Save path exists: ", day_path)
else:
    os.makedirs(day_path)
    print("Save path is created: ", day_path)


'''

cnt = 0
for i in tqdm(range(len_stocks)):
    df_tmp = df[symbols.iloc[i,0] == df['Code']]
    
    save_path = os.path.join("/home/DATA/ymh/s_modeling/data/cumulative3",symbols.iloc[i,0])
    
    mean_5 = Replay_Memory(5)              # 5일 평균 버퍼
    mean_10 = Replay_Memory(10)            # 10일 평균 버퍼
    mean_20 = Replay_Memory(20)            # 20일 평균 버퍼
    mean_60 = Replay_Memory(60)            # 60일 평균 버퍼
    mean_120 = Replay_Memory(120)          # 120일 평균 버퍼
        
    for j in range((len(df_tmp)-1),len(df_tmp)):
        oldday_path = os.path.join(save_path, '2021-06-11')
        if os.path.isdir(oldday_path):
            filename = os.listdir(oldday_path)
            filename.sort()
        
            ema = np.load(os.path.join(oldday_path, "EMA.npy"))
            mean_5.buffer[0:int(filename[1].split("_")[2]),:] = np.load(os.path.join(oldday_path, "mean_05_%d_%d_.npy" % (int(filename[1].split("_")[2]), int(filename[1].split("_")[3]))))
            mean_5.n_entries = int(filename[1].split("_")[2])
            mean_5.buffer_idx = int(filename[1].split("_")[3])
            mean_10.buffer[0:int(filename[2].split("_")[2]),:] = np.load(os.path.join(oldday_path, "mean_10_%d_%d_.npy" % (int(filename[2].split("_")[2]), int(filename[2].split("_")[3]))))
            mean_10.n_entries = int(filename[2].split("_")[2])
            mean_10.buffer_idx = int(filename[2].split("_")[3])
            mean_20.buffer[0:int(filename[4].split("_")[2]),:] = np.load(os.path.join(oldday_path, "mean_20_%d_%d_.npy" % (int(filename[4].split("_")[2]), int(filename[4].split("_")[3]))))
            mean_20.n_entries = int(filename[4].split("_")[2])
            mean_20.buffer_idx = int(filename[4].split("_")[3])
            mean_60.buffer[0:int(filename[5].split("_")[2]),:] = np.load(os.path.join(oldday_path, "mean_60_%d_%d_.npy" % (int(filename[5].split("_")[2]), int(filename[5].split("_")[3]))))
            mean_60.n_entries = int(filename[5].split("_")[2])
            mean_60.buffer_idx = int(filename[5].split("_")[3])
            mean_120.buffer[0:int(filename[3].split("_")[2]),:] = np.load(os.path.join(oldday_path, "mean_120_%d_%d_.npy" % (int(filename[3].split("_")[2]), int(filename[3].split("_")[3]))))
            mean_120.n_entries = int(filename[3].split("_")[2])
            mean_120.buffer_idx = int(filename[3].split("_")[3])
        
            newday_path = os.path.join(save_path, '2021-06-14')
            if os.path.isdir(newday_path):
                #print("Save path exists: ", day_path)
                pass
            else:
                os.makedirs(newday_path)

            Open = df_tmp.iloc[j,2]
            High = df_tmp.iloc[j,3]
            Low = df_tmp.iloc[j,4]
            Close = df_tmp.iloc[j,1]
            Volume = df_tmp.iloc[j,5]
        
            ema_Open = 0.01*Open + 0.99*ema[0,0]
            ema_High = 0.01*High + 0.99*ema[1,0]
            ema_Low = 0.01*Low + 0.99*ema[2,0]
            ema_Close = 0.01*Close + 0.99*ema[3,0]
            ema_Volume = 0.01*Volume + 0.99*ema[4,0]

            onedata = np.zeros((5,1))
                
            onedata[0,0] = ema_Open
            onedata[1,0] = ema_High
            onedata[2,0] = ema_Low
            onedata[3,0] = ema_Close
            onedata[4,0] = ema_Volume
            
            mean_5.add(Open, High, Low, Close, Volume)
            mean_10.add(Open, High, Low, Close, Volume)
            mean_20.add(Open, High, Low, Close, Volume)
            mean_60.add(Open, High, Low, Close, Volume)
            mean_120.add(Open, High, Low, Close, Volume)        
        
            name = os.path.join(newday_path,"EMA")
            np.save(name, onedata)
    
            name = os.path.join(newday_path,"mean_05_%d_%d_" % (mean_5.n_entries, mean_5.buffer_idx))
            np.save(name, mean_5.buffer[0:mean_5.n_entries,:])
            
            name = os.path.join(newday_path,"mean_10_%d_%d_" % (mean_10.n_entries, mean_10.buffer_idx))
            np.save(name, mean_10.buffer[0:mean_10.n_entries,:])
    
            name = os.path.join(newday_path,"mean_20_%d_%d_" % (mean_20.n_entries, mean_20.buffer_idx))
            np.save(name, mean_20.buffer[0:mean_20.n_entries,:])
        
            name = os.path.join(newday_path,"mean_60_%d_%d_" % (mean_60.n_entries, mean_60.buffer_idx))
            np.save(name, mean_60.buffer[0:mean_60.n_entries,:])
    
            name = os.path.join(newday_path,"mean_120_%d_%d_" % (mean_120.n_entries, mean_120.buffer_idx))
            np.save(name, mean_120.buffer[0:mean_120.n_entries,:])




















