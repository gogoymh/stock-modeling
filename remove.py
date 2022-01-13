from marcap import marcap_data
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import shutil

symbols = pd.read_csv("/home/DATA/ymh/s_modeling/data/data_2822_20210608.csv", encoding='utf-8')
symbols = symbols.rename(columns={"단축코드":"Code", "시장구분":"Market"})
symbols = pd.concat([symbols['Code'], symbols['Market']], axis=1)
symbols = symbols[symbols.Market != 'KONEX']

len_stocks = len(symbols) # 종목의 총 개수

for i in tqdm(range(len_stocks)):
    
    save_path = os.path.join("/home/DATA/ymh/s_modeling/data/cumulative3",symbols.iloc[i,0])
    
    newday_path = os.path.join(save_path, '2021-06-08')
    if os.path.isdir(newday_path):
        #os.rmdir(newday_path)
        shutil.rmtree(newday_path)
