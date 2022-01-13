#import FinanceDataReader as fdr

#import pandas as pd
from tqdm import tqdm
import numpy as np
#from datetime import date, timedelta
import os

#pd.set_option('display.max_columns', None)

#==========================================================================================================

#print(oneday)
#print(np.unique(np.where(oneday == 0)[0]))

# 시고저종
# 수익률 = ((고-시)/시)*100
# 라벨 가능성1: label = (수익률 + 60)/120 # 이론상 하루 최대 수익률은 전날 종가 대비 하한가에서 상한가에 도달한 것이다.
# 라벨 가능성2: 수익률 3% 이상인 종목에 대하여 1, 아니면 0으로 -> precision을 극대화 하면 된다.

# 0 번째 input의 output은 1번에서 만들어야 한다.

'''
input_path = "C:\\유민형\\개인 연구\\stock_modeling\\tmp"
output_path = "C:\\유민형\\개인 연구\\stock_modeling\\tmp"

i = 0
oneday = np.load(os.path.join(input_path,'input_%05d.npy' % (i+1)))

output = np.zeros((oneday.shape[0]))

for j in range(oneday.shape[0]):
    if oneday[j,0] != 0:
        profit_rate = ((oneday[j,3] - oneday[j,0])/oneday[j,0])*100
        
        #output[j] = (profit_rate + 60)/120
        
        if profit_rate >= 3:
            output[j] = 1
        else:
            output[j] = 0


np.save(os.path.join(output_path,'output_%05d' % i), output) # 0 번째 input의 output은 1번에서 만들어야 한다.
'''

input_path = "/home/compu/ymh/s_modeling/data/input_v1"
output_path = "/home/compu/ymh/s_modeling/data/output_v3"

for i in tqdm(range(len(os.listdir(input_path))-1)):
    oneday = np.load(os.path.join(input_path,'input_%05d.npy' % (i+1)))

    output = np.zeros((oneday.shape[0]))

    for j in range(oneday.shape[0]):
        if oneday[j,0] != 0:
            profit_rate = ((oneday[j,1] - oneday[j,0])/oneday[j,0])*100
            
            if profit_rate >= 5:
                output[j] = 1
            else:
                output[j] = 0

    np.save(os.path.join(output_path,'output_%05d' % i), output)
