import FinanceDataReader as fdr

import pandas as pd
from tqdm import tqdm
import numpy as np
#from datetime import date, timedelta
import os

#pd.set_option('display.max_columns', None)

#==========================================================================================================
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

# 전체 종목 목록
symbols = pd.read_pickle("symbols.pkl")

#symbols = pd.read_csv("symbols.csv")
#symbols = symbols.drop(['Unnamed: 0'], axis=1)

print(symbols)
len_stocks = len(symbols) # 종목의 총 개수

df_rep = fdr.DataReader(symbol='005930') # 거래일을 알기 위한 대표 종목: 삼성전자
today = str(df_rep.index.values[-1])[:10]
print(today)

# 모든 종목 모든 날짜 만들기
for i in tqdm(range(len_stocks)):
    df = fdr.DataReader(symbol=symbols[i]) # 종목 정보 불러오기
    mean_5 = Replay_Memory(5)              # 5일 평균 버퍼
    mean_10 = Replay_Memory(10)            # 10일 평균 버퍼
    mean_20 = Replay_Memory(20)            # 20일 평균 버퍼
    mean_60 = Replay_Memory(60)            # 60일 평균 버퍼
    mean_120 = Replay_Memory(120)          # 120일 평균 버퍼
    
    ema_Open = 0
    ema_High = 0
    ema_Low = 0
    ema_Close = 0
    ema_Volume = 0
    
    ema_start = 0
    
    start = len(df)
    
    if start >= 120:
        before = start - 120
    else:
        before = 0

    for j in range(before, (start-1)):
        Open = df.iloc[j,0]
        High = df.iloc[j,1]
        Low = df.iloc[j,2]
        Close = df.iloc[j,3]
        Volume = df.iloc[j,4]
            
        mean_5.add(Open, High, Low, Close, Volume)
        mean_10.add(Open, High, Low, Close, Volume)
        mean_20.add(Open, High, Low, Close, Volume)
        mean_60.add(Open, High, Low, Close, Volume)
        mean_120.add(Open, High, Low, Close, Volume)
            
        if ema_start == 0:
            ema_Open = Open
            ema_High = High
            ema_Low = Low
            ema_Close = Close
            ema_Volume = Volume
                
            ema_start = 1
                
        else:
            ema_Open = 0.01*Open + 0.99*ema_Open
            ema_High = 0.01*High + 0.99*ema_High
            ema_Low = 0.01*Low + 0.99*ema_Low
            ema_Close = 0.01*Close + 0.99*ema_Close
            ema_Volume = 0.01*Volume + 0.99*ema_Volume
                
            
    ## ---- Input ---- ##
    onedata = np.zeros((7, 5))
    
    Open = df.iloc[(start-1),0]
    High = df.iloc[(start-1),1]
    Low = df.iloc[(start-1),2]
    Close = df.iloc[(start-1),3]
    Volume = df.iloc[(start-1),4]
        
    mean_5.add(Open, High, Low, Close, Volume)
    mean_10.add(Open, High, Low, Close, Volume)
    mean_20.add(Open, High, Low, Close, Volume)
    mean_60.add(Open, High, Low, Close, Volume)
    mean_120.add(Open, High, Low, Close, Volume)
        
    if ema_start == 0:
        ema_Open = Open
        ema_High = High
        ema_Low = Low
        ema_Close = Close
        ema_Volume = Volume
                
        ema_start = 1
                
    else:
        ema_Open = 0.01*Open + 0.99*ema_Open
        ema_High = 0.01*High + 0.99*ema_High
        ema_Low = 0.01*Low + 0.99*ema_Low
        ema_Close = 0.01*Close + 0.99*ema_Close
        ema_Volume = 0.01*Volume + 0.99*ema_Volume
        
    ## 시가
    onedata[0,0] = Open
    onedata[1,0] = mean_5.get_mean()[0]
    onedata[2,0] = mean_10.get_mean()[0]
    onedata[3,0] = mean_20.get_mean()[0]
    onedata[4,0] = mean_60.get_mean()[0]
    onedata[5,0] = mean_120.get_mean()[0]
    onedata[6,0] = ema_Open
        
    ## 고가
    onedata[0,1] = High
    onedata[1,1] = mean_5.get_mean()[1]
    onedata[2,1] = mean_10.get_mean()[1]
    onedata[3,1] = mean_20.get_mean()[1]
    onedata[4,1] = mean_60.get_mean()[1]
    onedata[5,1] = mean_120.get_mean()[1]
    onedata[6,1] = ema_High
        
    ## 저가
    onedata[0,2] = Low
    onedata[1,2] = mean_5.get_mean()[2]
    onedata[2,2] = mean_10.get_mean()[2]
    onedata[3,2] = mean_20.get_mean()[2]
    onedata[4,2] = mean_60.get_mean()[2]
    onedata[5,2] = mean_120.get_mean()[2]
    onedata[6,2] = ema_Low
        
    ## 종가
    onedata[0,3] = Close
    onedata[1,3] = mean_5.get_mean()[3]
    onedata[2,3] = mean_10.get_mean()[3]
    onedata[3,3] = mean_20.get_mean()[3]
    onedata[4,3] = mean_60.get_mean()[3]
    onedata[5,3] = mean_120.get_mean()[3]
    onedata[6,3] = ema_Close
        
    ## 거래량
    onedata[0,4] = Volume
    onedata[1,4] = mean_5.get_mean()[4]
    onedata[2,4] = mean_10.get_mean()[4]
    onedata[3,4] = mean_20.get_mean()[4]
    onedata[4,4] = mean_60.get_mean()[4]
    onedata[5,4] = mean_120.get_mean()[4]
    onedata[6,4] = ema_Volume

    #name = os.path.join("/DATA/ymh/s_modeling/data/input_v2", "input_%08d_%d_" % (cnt, buy))
    #name = os.path.join("/home/compu/ymh/s_modeling/data/input_tmp","input_%08d_%d_" % (cnt, buy))
    #name = os.path.join("/data/ymh/s_modeling/data/input_v4","input_%08d_%d_%f_" % (cnt, buy, profit))
    #name = os.path.join("/home/DATA/ymh/s_modeling/data/input_v4","input_%08d_%d_%f_" % (cnt, buy, profit))
    name = os.path.join("/home/DATA/ymh/s_modeling/data/inference_v13_20210604","input_%08d" % i)
    np.save(name, onedata)

        




