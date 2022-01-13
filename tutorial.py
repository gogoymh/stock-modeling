import FinanceDataReader as fdr

import pandas as pd
from tqdm import tqdm
import numpy as np
#from datetime import date, timedelta
import os

#pd.set_option('display.max_columns', None)

#==========================================================================================================

# 주가 크롤링
df_kospi = fdr.StockListing('KOSPI') # NaN 존재: 인버스 같은 종목이나 우선주 종목은 이름 외 정보가 없음
df_kosdaq = fdr.StockListing('KOSDAQ') # NaN 존재: 홈페이지 주소가 없을 수도 있다

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

df_rep = fdr.DataReader(symbol='005930') # 거래일을 알기 위한 대표 종목: 삼성전자
#today = str(df_rep.index.values[-1])[:10] # 오늘 날짜: 장 종료 후 사용
#yesterday = str(df_rep.index.values[-2])[:10] # 어제 날짜: 장 진행 중 사용
'''
# 모든 종목 모든 날짜 만들기
for i in tqdm(range(len(df_rep))):
    date = str(df_rep.index.values[-(i+1)])[:10]
    oneday = np.zeros((len_stocks, 5))
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









