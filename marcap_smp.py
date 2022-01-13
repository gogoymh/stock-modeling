from marcap import marcap_data
import pandas as pd

df = marcap_data('2015-06-15', '2021-06-03', code='005930')
df = df.drop(['Name', 'Market', 'MarketId', 'Rank',
             'Dept', 'Marcap', 'ChangeCode', 'Changes',
             'ChagesRatio', 'Amount'], axis=1)
df['Date'] = df.index
df['DayOfWeek'] = df['Date'].dt.day_name()
df = df.drop(['Date'], axis=1)
df_dayofweek = pd.get_dummies(df['DayOfWeek'])

print(df.head())
print(df_dayofweek.head())