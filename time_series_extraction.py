import statsmodels.api as sm
from statsmodels.tsa.stattools import grangercausalitytests
import numpy as np
import pandas as pd

#load data and clean the column names
data = pd.read_csv("Iowa2000PresidentOdds.csv",sep=',')
data.columns = data.columns.str.strip().str.replace('\xa0\xa0\xa0\xa0', '')
#fill in missing value on price data and remove space in character data
data['AvgPrice'].fillna(data['LastPrice'], inplace=True)
data['Contract'] = data['Contract'].str.strip()

#extract only unique dates for Bush and Gore separately
Gore = pd.DataFrame()
Bush = pd.DataFrame()
Gore[['Date','Price']]=data[data['Contract']=='Dem'][['Date','AvgPrice']]
Bush[['Date','Price']]=data[data['Contract']=='Rep'][['Date','AvgPrice']]
Gore.reset_index(drop=True, inplace=True)
Bush.reset_index(drop=True, inplace=True)
#calculate the normalized probability of Gore winning the election
Gore['P_Gore_wins'] = Gore['Price']/(Gore['Price']+Bush["Price"])
Gore.drop('Price', inplace=True, axis=1)

#calculate 3-day moving average of the probability of Gore winning
for i in range(0,Gore.shape[0]-2):
    Gore.loc[Gore.index[i+2],'P_Gore_wins_3MA'] = ((Gore.iloc[i,1]+ Gore.iloc[i+1,1] +Gore.iloc[i+2,1])/3)

Gore.drop('P_Gore_wins', inplace=True, axis=1)
Gore.dropna(inplace=True)
Gore.reset_index(drop=True, inplace=True)
