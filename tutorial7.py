import FinanceDataReader as fdr

import pandas as pd
from tqdm import tqdm
import numpy as np
#from datetime import date, timedelta
import os

#pd.set_option('display.max_columns', None)

#==========================================================================================================

# 주가 크롤링
#df_kospi = fdr.StockListing('KOSPI') # NaN 존재: 인버스 같은 종목이나 우선주 종목은 이름 외 정보가 없음
#df_kosdaq = fdr.StockListing('KOSDAQ') # NaN 존재: 홈페이지 주소가 없을 수도 있다

#print("Crawling done.")

'''
## 상장 폐지 종목이 포함되어 있는지 여부 조사 # 포함 안 됨.
df_delist = fdr.StockListing('KRX-DELISTING')
df_admin = fdr.StockListing('KRX-ADMINISTRATIVE')

cnt = 0
for i in tqdm(range(len(df_admin))):
    if df_delist['Symbol'][i] in df_kosdaq['Symbol']:
        cnt += 1
    if df_delist['Symbol'][i] in df_kospi['Symbol']:
        cnt += 1
    
    #if df_admin['Symbol'][i] in df_kosdaq['Symbol']:
    #    cnt += 1
    #if df_admin['Symbol'][i] in df_kospi['Symbol']:
    #    cnt += 1
    
        
print(cnt)
'''

# 종가로 ploting 해보기
#df = fdr.DataReader(symbol='006840')
#df['Close'].plot()


# 종목명 데이터프레임 만들기: 내 데스크탑에선 30분이 소요된다. 미리 해서 저장해두자.
'''
symbols = pd.concat([df_kospi["Symbol"], df_kosdaq["Symbol"]], ignore_index=True)

for_drop = []
for i in tqdm(range(len(symbols))):
    try:
        df = fdr.DataReader(symbol=symbols[i])
        if df.empty:
            for_drop.append(i)
    except:
        for_drop.append(i)

symbols = symbols.drop(for_drop)
len_stocks = len(symbols) # 종목의 총 개수
print("종목 개수:%d" % len_stocks)

symbols = symbols.reset_index(drop=True)
symbols.to_pickle("symbols.pkl")
'''
symbols = pd.read_pickle("symbols.pkl")

#symbols = pd.read_csv("symbols.csv")
#symbols = symbols.drop(['Unnamed: 0'], axis=1)

print(symbols)
len_stocks = len(symbols) # 종목의 총 개수

'''
# 하루 단위의 numpy array를 만들기
oneday = np.zeros((len_stocks, 5))

# 첫번째 종목 채워넣기
df = fdr.DataReader(symbol=symbols[0]) # 첫번째 종목=symbols[0]

#today = str(date.today()) # 오늘 날짜: 장 종료 후 사용
#yesterday = str(date.today() - timedelta(1)) # 어제 날짜: 장 진행 중 사용

# 거래일의 정보가 담겨있어 행의 인덱스를 사용하는 것이 좋을 듯 하다.
today = str(df.index.values[-1])[:10] # 오늘 날짜: 장 종료 후 사용
yesterday = str(df.index.values[-2])[:10] # 어제 날짜: 장 진행 중 사용

oneday[0,0] = df.loc[yesterday, 'Open'] # 시가='Open'
oneday[0,1] = df.loc[yesterday, 'High'] # 고가='High'
oneday[0,2] = df.loc[yesterday, 'Low'] # 저가='Low'
oneday[0,3] = df.loc[yesterday, 'Close'] # 종가='Close'
oneday[0,4] = df.loc[yesterday, 'Volume'] # 거래량='Volume'
'''


'''
# 모든 종목 한 날짜에 채워넣기
oneday = np.zeros((len_stocks, 5))

df_rep = fdr.DataReader(symbol='005930') # 거래일을 알기 위한 대표 종목: 삼성전자
today = str(df_rep.index.values[-1])[:10] # 오늘 날짜: 장 종료 후 사용
yesterday = str(df_rep.index.values[-2])[:10] # 어제 날짜: 장 진행 중 사용

for i in tqdm(range(len_stocks)):
    df = fdr.DataReader(symbol=symbols[i])
    try:
        oneday[i,0] = df.loc[yesterday, 'Open'] # 시가='Open'
        oneday[i,1] = df.loc[yesterday, 'High'] # 고가='High'
        oneday[i,2] = df.loc[yesterday, 'Low'] # 저가='Low'
        oneday[i,3] = df.loc[yesterday, 'Close'] # 종가='Close'
        oneday[i,4] = df.loc[yesterday, 'Volume'] # 거래량='Volume'
    except:
        pass

np.save(yesterday, oneday)
'''

#oneday = np.load('2021-05-03.npy')
#print(oneday)
#print(np.unique(np.where(oneday == 0)[0]))



# 날짜 제한 가격제한폭 30% 시작일
last_date = '2015-06-15'

#df_rep = fdr.DataReader(symbol='005930') # 거래일을 알기 위한 대표 종목: 삼성전자
#today = str(df_rep.index.values[-1])[:10] # 오늘 날짜: 장 종료 후 사용
#yesterday = str(df_rep.index.values[-2])[:10] # 어제 날짜: 장 진행 중 사용

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


# 모든 종목 모든 날짜 만들기
cnt = 0
for i in tqdm(range(len_stocks)):
    df = fdr.DataReader(symbol=symbols[i]) # 종목 정보 불러오기    
    
    ema_start = 0
    
    start = 0
    for k in range(len(df)):
        if last_date == str(df.index.values[k])[:10]:
            start = k
            break
    
    if start > 0:
        if start > 120:
            before = start - 120
        else:
            before = 0
        
        for j in range(before, start):
            Open = df.iloc[j,0]
            High = df.iloc[j,1]
            Low = df.iloc[j,2]
            Close = df.iloc[j,3]
            Volume = df.iloc[j,4]
                        
            if ema_start == 0:
                ema_1_Open = Open
                ema_1_High = High
                ema_1_Low = Low
                ema_1_Close = Close
                ema_1_Volume = Volume
                
                ema_3_Open = Open
                ema_3_High = High
                ema_3_Low = Low
                ema_3_Close = Close
                ema_3_Volume = Volume
                
                ema_5_Open = Open
                ema_5_High = High
                ema_5_Low = Low
                ema_5_Close = Close
                ema_5_Volume = Volume
                
                ema_7_Open = Open
                ema_7_High = High
                ema_7_Low = Low
                ema_7_Close = Close
                ema_7_Volume = Volume
                
                ema_9_Open = Open
                ema_9_High = High
                ema_9_Low = Low
                ema_9_Close = Close
                ema_9_Volume = Volume
                
                ema_start = 1
                
            else:
                ema_1_Open = 0.1*Open + 0.9*ema_1_Open
                ema_1_High = 0.1*High + 0.9*ema_1_High
                ema_1_Low = 0.1*Low + 0.9*ema_1_Low
                ema_1_Close = 0.1*Close + 0.9*ema_1_Close
                ema_1_Volume = 0.1*Volume + 0.9*ema_1_Volume
                
                ema_3_Open = 0.3*Open + 0.7*ema_3_Open
                ema_3_High = 0.3*High + 0.7*ema_3_High
                ema_3_Low = 0.3*Low + 0.7*ema_3_Low
                ema_3_Close = 0.3*Close + 0.7*ema_3_Close
                ema_3_Volume = 0.3*Volume + 0.7*ema_3_Volume
                
                ema_5_Open = 0.5*Open + 0.5*ema_5_Open
                ema_5_High = 0.5*High + 0.5*ema_5_High
                ema_5_Low = 0.5*Low + 0.5*ema_5_Low
                ema_5_Close = 0.5*Close + 0.5*ema_5_Close
                ema_5_Volume = 0.5*Volume + 0.5*ema_5_Volume
                
                ema_7_Open = 0.7*Open + 0.3*ema_7_Open
                ema_7_High = 0.7*High + 0.3*ema_7_High
                ema_7_Low = 0.7*Low + 0.3*ema_7_Low
                ema_7_Close = 0.7*Close + 0.3*ema_7_Close
                ema_7_Volume = 0.7*Volume + 0.3*ema_7_Volume
            
                ema_9_Open = 0.9*Open + 0.1*ema_9_Open
                ema_9_High = 0.9*High + 0.1*ema_9_High
                ema_9_Low = 0.9*Low + 0.1*ema_9_Low
                ema_9_Close = 0.9*Close + 0.1*ema_9_Close
                ema_9_Volume = 0.9*Volume + 0.1*ema_9_Volume
            
    for j in range(start, (len(df)-5)):
        ## ---- Output ---- ##
        if df.iloc[(j+1),0] == 0:
            buy = 0
            profit = 0
            stop = 0
        else:
            ## 오늘 시가 대비 앞으로 3일내 고가 5% 이상 상승
            profit_1 = ((df.iloc[(j+1),1] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 오늘 고가 오늘 시가 대비
            profit_2 = ((df.iloc[(j+2),1] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 내일 고가 오늘 시가 대비
            profit_3 = ((df.iloc[(j+3),1] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 모레 고가 오늘 시가 대비
            #profit_4 = ((df.iloc[(j+4),1] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 
            #profit_5 = ((df.iloc[(j+5),1] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 
            
            stop_1 = ((df.iloc[(j+1),2] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 오늘 저가 오늘 시가 대비
            stop_2 = ((df.iloc[(j+2),2] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 내일 저가 오늘 시가 대비
            stop_3 = ((df.iloc[(j+3),2] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 모레 저가 오늘 시가 대비
            #stop_4 = ((df.iloc[(j+4),2] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 모레 저가 오늘 시가 대비
            #stop_5 = ((df.iloc[(j+4),2] - df.iloc[(j+1),0])/df.iloc[(j+1),0])*100 # 모레 저가 오늘 시가 대비
            
            stop_12 = min(stop_1, stop_2)
            stop_123 = min(stop_1, stop_2, stop_3)
            #stop_1234 = min(stop_1, stop_2, stop_3, stop_4)
            #stop_12345 = min(stop_1, stop_2, stop_3, stop_4, stop_5)
            
            if profit_1 >= 5:
                if stop_1 >= -2:
                    buy = 1
                else:
                    buy = 0
                profit = profit_1
                stop = stop_1
            elif profit_2 >= 5:
                if stop_12 >= -2:
                    buy = 1
                else:
                    buy = 0
                profit = profit_2
                stop = stop_12
            elif profit_3 >= 5:
                if stop_123 >= -2:
                    buy = 1
                else:
                    buy = 0
                profit = profit_3
                stop = stop_123
            else:
                buy = 0
                profit = max(profit_1, profit_2, profit_3)#, profit_4, profit_5)
                stop = stop_123#stop_12345
            '''
            elif profit_4 >= 2:
                if (profit_4+stop_1234) >= 2:
                    buy = 1
                elif stop_1234 >= -1.5:
                    buy = 1                
                else:
                    buy = 0
                profit = profit_4
                stop = stop_1234
            elif profit_5 >= 2:
                if (profit_5+stop_12345) >= 2:
                    buy = 1
                elif stop_12345 >= -1.5:
                    buy = 1
                else:
                    buy = 0
                profit = profit_5
                stop = stop_12345
            '''
            
                        
        ## ---- Input ---- ##
        onedata = np.zeros((6, 5))
        
        Open = df.iloc[j,0]
        High = df.iloc[j,1]
        Low = df.iloc[j,2]
        Close = df.iloc[j,3]
        Volume = df.iloc[j,4]
        
        if ema_start == 0:
            ema_1_Open = Open
            ema_1_High = High
            ema_1_Low = Low
            ema_1_Close = Close
            ema_1_Volume = Volume
            
            ema_3_Open = Open
            ema_3_High = High
            ema_3_Low = Low
            ema_3_Close = Close
            ema_3_Volume = Volume
            
            ema_5_Open = Open
            ema_5_High = High
            ema_5_Low = Low
            ema_5_Close = Close
            ema_5_Volume = Volume
            
            ema_7_Open = Open
            ema_7_High = High
            ema_7_Low = Low
            ema_7_Close = Close
            ema_7_Volume = Volume
            
            ema_9_Open = Open
            ema_9_High = High
            ema_9_Low = Low
            ema_9_Close = Close
            ema_9_Volume = Volume
            
            ema_start = 1
                
        else:
            ema_1_Open = 0.1*Open + 0.9*ema_1_Open
            ema_1_High = 0.1*High + 0.9*ema_1_High
            ema_1_Low = 0.1*Low + 0.9*ema_1_Low
            ema_1_Close = 0.1*Close + 0.9*ema_1_Close
            ema_1_Volume = 0.1*Volume + 0.9*ema_1_Volume
            
            ema_3_Open = 0.3*Open + 0.7*ema_3_Open
            ema_3_High = 0.3*High + 0.7*ema_3_High
            ema_3_Low = 0.3*Low + 0.7*ema_3_Low
            ema_3_Close = 0.3*Close + 0.7*ema_3_Close
            ema_3_Volume = 0.3*Volume + 0.7*ema_3_Volume
        
            ema_5_Open = 0.5*Open + 0.5*ema_5_Open
            ema_5_High = 0.5*High + 0.5*ema_5_High
            ema_5_Low = 0.5*Low + 0.5*ema_5_Low
            ema_5_Close = 0.5*Close + 0.5*ema_5_Close
            ema_5_Volume = 0.5*Volume + 0.5*ema_5_Volume
            
            ema_7_Open = 0.7*Open + 0.3*ema_7_Open
            ema_7_High = 0.7*High + 0.3*ema_7_High
            ema_7_Low = 0.7*Low + 0.3*ema_7_Low
            ema_7_Close = 0.7*Close + 0.3*ema_7_Close
            ema_7_Volume = 0.7*Volume + 0.3*ema_7_Volume
            
            ema_9_Open = 0.9*Open + 0.1*ema_9_Open
            ema_9_High = 0.9*High + 0.1*ema_9_High
            ema_9_Low = 0.9*Low + 0.1*ema_9_Low
            ema_9_Close = 0.9*Close + 0.1*ema_9_Close
            ema_9_Volume = 0.9*Volume + 0.1*ema_9_Volume
            
        ## 시가
        onedata[0,0] = Open
        onedata[1,0] = ema_1_Open
        onedata[2,0] = ema_3_Open
        onedata[3,0] = ema_5_Open
        onedata[4,0] = ema_7_Open
        onedata[5,0] = ema_9_Open
        
        ## 고가
        onedata[0,1] = High
        onedata[1,1] = ema_1_High
        onedata[2,1] = ema_3_High
        onedata[3,1] = ema_5_High
        onedata[4,1] = ema_7_High
        onedata[5,1] = ema_9_High
        
        ## 저가
        onedata[0,2] = Low
        onedata[1,2] = ema_1_Low
        onedata[2,2] = ema_3_Low
        onedata[3,2] = ema_5_Low
        onedata[4,2] = ema_7_Low
        onedata[5,2] = ema_9_Low
        
        ## 종가
        onedata[0,3] = Close
        onedata[1,3] = ema_1_Close
        onedata[2,3] = ema_3_Close
        onedata[3,3] = ema_5_Close
        onedata[4,3] = ema_7_Close
        onedata[5,3] = ema_9_Close
        
        ## 거래량
        onedata[0,4] = Volume
        onedata[1,4] = ema_1_Volume
        onedata[2,4] = ema_3_Volume
        onedata[3,4] = ema_5_Volume
        onedata[4,4] = ema_7_Volume
        onedata[5,4] = ema_9_Volume

        #name = os.path.join("/DATA/ymh/s_modeling/data/input_v2", "input_%08d_%d_" % (cnt, buy))
        #name = os.path.join("/home/compu/ymh/s_modeling/data/input_tmp","input_%08d_%d_" % (cnt, buy))
        #name = os.path.join("/data/ymh/s_modeling/data/input_v4","input_%08d_%d_%f_" % (cnt, buy, profit))
        name = os.path.join("/home/DATA/ymh/s_modeling/data/input_v12","input_%08d_%d_%f_%f_" % (cnt, buy, profit, stop))
        np.save(name, onedata)
        
        #print(i,j,cnt)
        cnt += 1
        


'''
for i in tqdm(range(len(df_rep))):
    date = str(df_rep.index.values[-(i+1)])[:10]
    onedata = np.zeros((6, 5))
    for j in range(len_stocks):
        df = fdr.DataReader(symbol=symbols[j])
        try:
            oneday[j,0] = df.loc[date, 'Open'] # 시가='Open'
            oneday[j,1] = df.loc[date, 'High'] # 고가='High'
            oneday[j,2] = df.loc[date, 'Low'] # 저가='Low'
            oneday[j,3] = df.loc[date, 'Close'] # 종가='Close'
            oneday[j,4] = df.loc[date, 'Volume'] # 거래량='Volume'
        except:
            pass
    
    name = os.path.join("/DATA/ymh/s_modeling/data/input_v1","input_%05d" % i)
    #name = os.path.join("/home/compu/ymh/s_modeling/data/input_v1","input_%05d" % i)
    np.save(name, oneday)
    print(date)
'''

