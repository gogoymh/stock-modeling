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


symbols = pd.read_pickle("symbols.pkl")

print(symbols)
len_stocks = len(symbols) # 종목의 총 개수

cnt = 0
for i in tqdm(range(len_stocks)):
    df = marcap_data('2015-06-15', '2021-06-04', code=symbols[i])
    df = df.drop(['Name', 'Market', 'MarketId', 'Rank',
                  'Dept', 'Marcap', 'ChangeCode', 'Changes',
                  'ChagesRatio', 'Amount'], axis=1)
    df['Date'] = df.index
    df['DayOfWeek'] = df['Date'].dt.day_name()
    df = df.drop(['Date'], axis=1)
    df_dayofweek = pd.get_dummies(df['DayOfWeek'])

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

    for j in range(len(df)-3):
        onedata = np.zeros((41,1))
        
        Open = df.iloc[j,2]
        High = df.iloc[j,3]
        Low = df.iloc[j,4]
        Close = df.iloc[j,1]
        Volume = df.iloc[j,5]
        
        Stocks = df.iloc[j,6]
        Monday = df_dayofweek.iloc[j,1]
        Tuesday = df_dayofweek.iloc[j,3]
        Wednesday = df_dayofweek.iloc[j,4]
        Thursday = df_dayofweek.iloc[j,2]
        Friday = df_dayofweek.iloc[j,0]

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

        onedata[0,0] = Open
        onedata[1,0] = High
        onedata[2,0] = Low
        onedata[3,0] = Close
        onedata[4,0] = Volume
        
        onedata[5:10,0] = mean_5.get_mean()
        onedata[10:15,0] = mean_10.get_mean()
        onedata[15:20,0] = mean_20.get_mean()
        onedata[20:25,0] = mean_60.get_mean()
        onedata[25:30,0] = mean_120.get_mean()
        
        onedata[30,0] = ema_Open
        onedata[31,0] = ema_High
        onedata[32,0] = ema_Low
        onedata[33,0] = ema_Close
        onedata[34,0] = ema_Volume
        
        onedata[35,0] = Stocks
        onedata[36,0] = Monday
        onedata[37,0] = Tuesday
        onedata[38,0] = Wednesday
        onedata[39,0] = Thursday
        onedata[40,0] = Friday

        ##
        if df.iloc[(j+1),2] == 0:
            buy = 0
            profit = 0
            stop = 0
        else:
            ## 오늘 시가 대비 앞으로 3일내 고가 5% 이상 상승
            profit_1 = ((df.iloc[(j+1),3] - df.iloc[(j+1),2])/df.iloc[(j+1),2])*100 # 오늘 고가 오늘 시가 대비
            profit_2 = ((df.iloc[(j+2),3] - df.iloc[(j+1),2])/df.iloc[(j+1),2])*100 # 내일 고가 오늘 시가 대비
            profit_3 = ((df.iloc[(j+3),3] - df.iloc[(j+1),2])/df.iloc[(j+1),2])*100 # 모레 고가 오늘 시가 대비
            #profit_4 = ((df.iloc[(j+4),1] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 
            #profit_5 = ((df.iloc[(j+5),1] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 
            
            stop_1 = ((df.iloc[(j+1),1] - df.iloc[(j+1),2])/df.iloc[(j+1),2])*100 # 오늘 종가 오늘 시가 대비
            stop_2 = ((df.iloc[(j+2),1] - df.iloc[(j+1),2])/df.iloc[(j+1),2])*100 # 내일 종가 오늘 시가 대비
            stop_3 = ((df.iloc[(j+3),1] - df.iloc[(j+1),2])/df.iloc[(j+1),2])*100 # 모레 종가 오늘 시가 대비
            #stop_4 = ((df.iloc[(j+4),2] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 모레 종가 오늘 시가 대비
            #stop_5 = ((df.iloc[(j+4),2] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 모레 종가 오늘 시가 대비
            
            #stop_12 = min(stop_1, stop_2)
            #stop_123 = min(stop_1, stop_2, stop_3)
            #stop_1234 = min(stop_1, stop_2, stop_3, stop_4)
            #stop_12345 = min(stop_1, stop_2, stop_3, stop_4, stop_5)
            
            if profit_1 >= 5:
                buy = 1
                profit = profit_1
                stop = stop_1

            elif profit_2 >= 5:
                buy = 1
                profit = profit_2
                stop = stop_2
            elif profit_3 >= 5:
                buy = 1
                profit = profit_3
                stop = stop_3
            else:
                buy = 0
                profit = max(profit_1, profit_2, profit_3)
                stop = stop_3
            
        
        name = os.path.join("/home/DATA/ymh/s_modeling/data/input_v18","input_%08d_%d_%f_%f_" % (cnt, buy, profit, stop))
        np.save(name, onedata)
        
        #print(i,j,cnt)
        cnt += 1







