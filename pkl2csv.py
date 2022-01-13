import pandas as pd

#symbols = pd.read_pickle("symbols.pkl")

symbols = pd.read_csv("symbols.csv")
symbols = symbols.drop(['Unnamed: 0'], axis=1)
print(symbols.loc[1050,:])

#symbols.to_pickle("symbols.pkl")
#symbols.to_csv("symbols.csv")