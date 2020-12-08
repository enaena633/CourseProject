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
    Gore.loc[Gore.index[i+1],'P_Gore_wins_MA'] = ((Gore.iloc[i,1]+ Gore.iloc[i+1,1] +Gore.iloc[i+2,1])/3)
Gore.loc[Gore.index[0],'P_Gore_wins_MA'] = ((Gore.iloc[0,1] + Gore.iloc[1,1])/2)
Gore.loc[Gore.index[-1],'P_Gore_wins_MA'] = ((Gore.iloc[-1,1] + Gore.iloc[-2,1])/2)
Gore.drop('P_Gore_wins', inplace=True, axis=1)

'''  Granger Test '''
#input: 
#time_series_data is 182x2 in dimension with first column being date
#text_count_stream should be topic count across documents or word count across documents under a significant topic - 182x1 dimension
def granger_test(time_series_data, text_count_stream):
    time_series_data['count'] = text_count_stream
    data = np.diff(time_series_data.iloc[:,1:], axis = 0)

    #run granger test with maxlag of 5 days
    granger_test_result = grangercausalitytests(data, 5, addconst=True, verbose=False)

    #get the optimal lag based on the highest F-test value
    optimal_lag = -1
    F_test = -1.0
    for key in granger_test_result.keys():
        _F_test_ = granger_test_result[key][0]['params_ftest'][0]
        if _F_test_ > F_test:
            F_test = _F_test_
            optimal_lag = key

    #use the optimal lag to get the OLS estimate 
    #under the unrestricted model - this includes leads and lags
    #define impact value as the average coefficient of X's and check the sign
    significance = np.round(1 - granger_test_result[optimal_lag][0]['params_ftest'][1], 2)
    X_coeff_lst = granger_test_result[optimal_lag][1][1].params[:-1]
    if sum(X_coeff_lst)/len(X_coeff_lst) > 0:
        i_impact = 1
    else:
        i_impact = -1
    #output the sign of the impact value and significance value of the word
    sigs = np.multiply(significance, i_impact)
    return sigs

