import numpy as np
import pandas as pd
import math
import pymysql
import pdblp
import requests

from datetime import datetime
from datetime import timedelta
from datetime import date
from calendar import monthrange


FORECAST_DICT = {
     'NY No.11':[],
    'Hydrous Ethanol':[],
    'Anhydrous Ethanol':[],
    'Fertilizer Costs':[],
    'Brent Crude':[],
    'USDBRL':[],
    'Energy Prices':[],
    'SBMAR1 Comdty':[],
    'SBMAY1 Comdty':[],
    'SBJUL1 Comdty':[],
    'SBOCT1 Comdty':[],
    'SBMAR2 Comdty':[]
}
#Grabs Market Forecasts from Hedgebot Server API
FORECASTS = requests.get('http://3.142.53.5:8000/api/market_forecast_api/', params={'forecast_dict':FORECAST_DICT})


def attach_market_forecast(data_ls):
    """
    Takes the market forecasts from the hedgebot api and adds them to Monte Carlo input data depending on the 
    variable and dates referenced in input variable data_ls
    """
    date_df = pd.DataFrame(data_ls, columns = ['Date'])
    date_df['Date'] = pd.to_datetime(date_df['Date'], format= "%d/%m/%Y")
    forecasts_df = 1
    temp_ls = []
    for index, row in date_df.iterrows():
        quarter = row['Date'].quarter
        year = row['Date'].year

        if quarter == 1:
            month = 3
        elif quarter == 2:
            month = 6
        elif quarter == 3:
            month = 9
        elif quarter == 4:
            month = 12
        else:
            print('Error on assigning date to quarter')

        num_days = monthrange(year, month)[1]
        temp_date = pd.Timestamp(year = year, month = month, day = num_days)
        temp_ls.append(temp_date.strftime('%d/%m/%Y'))
    date_df['Forecast Date'] = temp_ls
    final_column_ls = ['USDBRL','Energy Prices', 'Brent Crude','USD Risk Free Rate','Brazil Central Bank Rate','Fertilizer Costs','NY No.11','Anhydrous Ethanol','Hydrous Ethanol']
    column_ls = forecasts_df['Description'].drop_duplicates().to_list()
    temp_ls_forecasts = []
    final_df = []
    forecasts_df['Date_published'] = pd.to_datetime(forecasts_df['Date_published'], format='%Y-%m-%d')
    for index, row in date_df.iterrows():
        temp_vals = forecasts_df.loc[forecasts_df['Date_published'] < row['Date']]
        temp_row = []
        for z in range(len(column_ls)):
            x = (temp_vals.loc[(temp_vals['Description'].values == column_ls[z])])  #) & (temp_vals['date_published'] == temp_vals['date_published'].max())])
            x_temp = pd.DataFrame(x.loc[x['Date_published'] == x['Date_published'].max()])
            #dates_df_period = x['forecast_period'].values[0]
            required_date = datetime.strptime(row['Forecast Date'], "%d/%m/%Y")
            x_temp['Forecast_period'] = pd.to_datetime(x['Forecast_period'], format = "%Y-%m-%d")
            x = x_temp.loc[x_temp['Forecast_period'] == required_date]

            diff = required_date.date() - row['Date'].date()
            diff = int(diff.days / 7)
            
            try:
                temp_vals_final = x['Forecasted_value'].values[0]
                temp_row.append(temp_vals_final)
            except:
                x = x_temp.loc[x_temp['Forecast_period'] == x_temp['Forecast_period'].min()]
                temp_row.append(x['Forecasted_value'].values[0])

        if len(final_df) < 1:
            final_df = pd.DataFrame([temp_row], columns = final_column_ls)
            #final_df['Time Difference (Weeks)'] = diff
        else:
            temp_df = pd.DataFrame([temp_row], columns= final_column_ls)
            #final_df['Time Difference (Weeks)'] = diff
            final_df = pd.concat([final_df, temp_df])
    
    final_df.index = data_ls
    return final_df
 
class GeometricBrownianMotion:
    """
    This class acts as a Monte Carlo simulation
    A new object is created for each monte carlo simulation
    The simulations are calculated through the simulate_paths function
    """
    def simulate_paths(self):
        while(self.T - self.dt > 0):
            dWt = np.random.normal(0, math.sqrt(self.dt))  # Brownian motion
            dYt = self.drift*self.dt + self.volatility*dWt  # Change in price
            self.current_price += self.current_price*dYt  # Add the change to the current price
            self.prices.append(self.current_price)  # Append new price to series
            self.T -= self.dt  # Account for the step in time
            
    #__init__ is an automated function that is called when object is created
    def __init__(self, initial_price, drift, volatility, dt, T):
        self.current_price = initial_price
        self.initial_price = initial_price
        self.drift = drift
        self.volatility = volatility
        self.dt = dt
        self.T = T
        self.prices = []
        self.simulate_paths()

        
def add_variable_path(data_df, day_duration, forecast_value):
    """
    This function is called by the main function (mc_test_new) and creates a GeometricBrownianMotion object for the given variable
    This function first calculates the required metrics for the Monte Carlo simulation (initial_price, drift, volatility, dt, T)
    then sends the variables to GeometricBrownianMotion which returns the path or Monte Carlo simulation data
    
    The input for this function includes data_df (the input data for the different market variables), day_duration (days to simulation), forecast_value
    (the forecasts for the market variable)
 
    """
    data = data_df
    data.columns = ['price']
    data['pct_chg'] = data.price.pct_change().fillna(0)
    data['log_ret'] = (np.log(data.price) - np.log(data.price.shift(1))).fillna(0)
    paths = 1000
    initial_price = data.price.iat[-1]
    drift = (forecast_value - initial_price) / initial_price
    volatility =  data.log_ret.std()
    dt = 1/day_duration
    T = 1
    price_paths = []

    for i in range(0, paths):
        price_paths.append(GeometricBrownianMotion(initial_price, drift, volatility, dt, T).prices)

    pct_chg_paths = pd.DataFrame(price_paths).T.pct_change().fillna(0)

    return price_paths, pct_chg_paths

def rescale_prices(data_df, initial_price):
    """
    This function rescales the prices after the Monte Carlo simulation based on the initial price (value of variable at day_0 of simulation)
    """
    data_df.iloc[0, :] *= initial_price
    for i in range(1, data_df.shape[0]):
        data_df.iloc[i, :] = data_df.iloc[i-1, :] * data_df.iloc[i, :]
    return data_df

def format_upload_new_data(data_df):
    """
    This function reformats the aggregated output data so that it is compatible with Hedgebot Server API database
    """
    temp_date_index = data_df['date']
    start_date = min(temp_date_index)
    end_date = max(temp_date_index)
    date_range = pd.date_range(start = start_date, end = end_date, freq='W-FRI')
    new_data_df = data_df.loc[data_df['date'].isin(date_range)]
    int_list = np.arange(0,1000)
    temp_row_ls = []
    for row, items in new_data_df.iterrows():
        forecast_period = items['date']
        reference = items['reference']
        t_row = items[int_list]
        temp_mu = t_row.mean()
        temp_sigma = t_row.std()
        sim_date = start_date
        end_date = end_date
        temp_row_ls.append({'forecast_period':forecast_period, 'reference':reference, 'returned_mean':temp_mu, 'returned_std':temp_sigma, 'sim_date':start_date, 'end_date':end_date})
    
    return temp_row_ls

def mc_test_new(initial_date, end_date, data_df):
    """
    This is the function that runs entire Monte Carlo Simulation
    The input is called by user and includes initial_date (day_0 of sim), end_date (day_n of sim) and data_df (sim input data)
    """
    
    """
    Stores backup list of dates and calcs needed date references for simulation
    By the end of this function, the method has stored a serialized / scaled dataframe of input data for simulation
    """
    date_range = pd.date_range(start = initial_date, end = end_date, freq='D')
    temp = data_df
    temp['Date'] = pd.to_datetime(temp['Date'])
    temp = temp.loc[temp['Date'] <= initial_date]
    diff = end_date - initial_date
    diff_days = len(date_range) 
    data_initial_date = initial_date-diff
    temp = temp.loc[temp['Date'] >= data_initial_date]
    temp =temp.sort_index(axis = 0, ascending = False)
    temp = temp.drop(['Date'], axis = 1)
    temp = temp.dropna()

    corr = np.corrcoef(temp, rowvar = False) #Cholesky matrix
    chol = np.linalg.cholesky(corr) #Cholesky matrix

    pct_chg_ls =[]
    price_path_ls = []
    initial_prices = {}
    """
    This section runs the Monte Carlo simulations based on the amount of input variables in temp (dataframe)
    By the end of this section, the code has stored simulation data into the empty lists initiated above
    """
    for i in range(len(temp.columns)):
        temp_df = temp.loc[:,[temp.columns[i]]]
        forecast_value = FORECASTS[temp.columns[i]][end_date]
        price_paths, pct_chg_paths = add_variable_path(temp_df, diff_days, forecast_value)
        price_paths = pd.DataFrame(price_paths)
        initial_prices[temp.columns[i]] = price_paths[0].iat[0]

        pct_chg_ls.append(pct_chg_paths)
        price_path_ls.append(price_paths)
        length = pct_chg_paths.shape[1]

    sugar_ls_final = []
    hydrous_ls_final = []
    anhydrous_ls_final = []
    fert_ls_final = []
    brent_ls_final = []
    usdbrl_ls_final = []
    energy_ls_final = []
    mar1_ls_final = []
    may1_ls_final = []
    jul1_ls_final = []
    oct1_ls_final = []
    mar2_ls_final = []
    """
    This section transforms the Monte Carlo output data with the cholesky matrix 
    By the end of this section, the code has stored final out data into the empty lists initiated above
    """
    labels = temp.columns
    for x in range(length):
        df_3 = pd.DataFrame()
        for q in range(len(pct_chg_ls)):
            df_3[labels[q]] = pct_chg_ls[q].iloc[:,x]
        corr_ret = pd.DataFrame(np.matmul(chol, df_3.to_numpy().T), index = labels).T #This line takes the cholesky matrix and applies it to df_3 (temp dataframe of sim output data)
        sugar_ls_final.append(corr_ret['NY No.11'].values)
        hydrous_ls_final.append(corr_ret['Hydrous Ethanol'].values)
        anhydrous_ls_final.append(corr_ret['Anhydrous Ethanol'].values)
        fert_ls_final.append(corr_ret['Fertilizer Costs'].values)
        brent_ls_final.append(corr_ret['Brent Crude'].values)
        usdbrl_ls_final.append(corr_ret['USDBRL'].values)
        energy_ls_final.append(corr_ret['Energy Prices'].values)
        mar1_ls_final.append(corr_ret['SBMAR1 Comdty'].values)
        may1_ls_final.append(corr_ret['SBMAY1 Comdty'].values)
        jul1_ls_final.append(corr_ret['SBJUL1 Comdty'].values)
        oct1_ls_final.append(corr_ret['SBOCT1 Comdty'].values)
        mar2_ls_final.append(corr_ret['SBMAR2 Comdty'].values)
    
    """
    Stores rescaled final output data in dataframes
    """
    sugar_df = pd.DataFrame(sugar_ls_final).T.apply(lambda x: x + 1)
    hydrous_df = pd.DataFrame(hydrous_ls_final).T.apply(lambda x: x + 1)
    anhydrous_df = pd.DataFrame(anhydrous_ls_final).T.apply(lambda x: x + 1)
    fert_df = pd.DataFrame(fert_ls_final).T.apply(lambda x: x + 1)
    brent_df = pd.DataFrame(brent_ls_final).T.apply(lambda x: x + 1)
    usdbrl_df = pd.DataFrame(usdbrl_ls_final).T.apply(lambda x: x + 1)
    energy_df = pd.DataFrame(energy_ls_final).T.apply(lambda x: x + 1)
    mar1_df = pd.DataFrame(mar1_ls_final).T.apply(lambda x: x + 1)
    may1_df = pd.DataFrame(may1_ls_final).T.apply(lambda x: x + 1)
    jul1_df = pd.DataFrame(jul1_ls_final).T.apply(lambda x: x + 1)
    oct1_df = pd.DataFrame(oct1_ls_final).T.apply(lambda x: x + 1)
    mar2_df = pd.DataFrame(mar2_ls_final).T.apply(lambda x: x + 1)

    """
    Rescales prices based on rescale_prices method (i.e. transforms them back into independant value paths
    """
    sugar_df_final = rescale_prices(sugar_df, initial_prices['NY No.11'])
    hydrous_df_final = rescale_prices(hydrous_df, initial_prices['Hydrous Ethanol'])
    anhydrous_df_final = rescale_prices(anhydrous_df, initial_prices['Anhydrous Ethanol'])
    fert_df_final = rescale_prices(fert_df, initial_prices['Fertilizer Costs'])
    brent_df_final = rescale_prices(brent_df, initial_prices['Brent Crude'])
    usdbrl_df_final = rescale_prices(usdbrl_df, initial_prices['USDBRL'])
    energy_df_final =rescale_prices(energy_df, initial_prices['Energy Prices'])
    mar1_df_final = rescale_prices(mar1_df, initial_prices['SBMAR1 Comdty'])
    may1_df_final = rescale_prices(may1_df, initial_prices['SBMAY1 Comdty'])
    jul1_df_final = rescale_prices(jul1_df, initial_prices['SBJUL1 Comdty'])
    oct1_df_final = rescale_prices(oct1_df, initial_prices['SBOCT1 Comdty'])
    mar2_df_final = rescale_prices(mar2_df, initial_prices['SBMAR2 Comdty'])

    """
    More dataframe management
    """
    sugar_df_final['reference'] = 'sugar_1'
    hydrous_df_final['reference'] = 'hydrous'
    anhydrous_df_final['reference'] = 'anhydrous'
    fert_df_final['reference'] = 'fert'
    brent_df_final['reference'] = 'brent'
    usdbrl_df_final['reference'] = 'usdbrl'
    energy_df_final['reference'] = 'energy'
    mar1_df_final['reference'] = 'sugar_mar1'
    may1_df_final['reference'] = 'sugar_may1'
    jul1_df_final['reference'] = 'sugar_jul1'
    oct1_df_final['reference'] = 'sugar_oct1'
    mar2_df_final['reference'] = 'sugar_mar2'

    """
    Adds dates back to final output dataframes
    """
    date_range = pd.date_range(start = initial_date, end = end_date, freq='D')
    if date_range.shape[0] > sugar_df_final.shape[0]:
        diff_range = date_range.shape[0] - sugar_df_final.shape[0]
        date_range = date_range.delete(diff_range)
    sugar_df_final['date'] = date_range
    hydrous_df_final['date'] = date_range
    anhydrous_df_final['date'] = date_range
    fert_df_final['date'] = date_range
    brent_df_final['date'] = date_range
    usdbrl_df_final['date'] = date_range
    energy_df_final['date'] = date_range
    mar1_df_final['date'] = date_range
    may1_df_final['date'] = date_range
    jul1_df_final['date'] = date_range
    oct1_df_final['date'] = date_range
    mar2_df_final['date'] = date_range

    final_df_ls = [sugar_df_final, hydrous_df_final, anhydrous_df_final, fert_df_final, brent_df_final, usdbrl_df_final, energy_df_final, mar1_df_final, may1_df_final, jul1_df_final, oct1_df_final, mar2_df_final]
    final_dict = {}
    for item in final_df_ls:
        temp_dict = format_upload_new_data(item)
        temp_label = item['reference'].iat[0]
        final_dict[temp_label] = temp_dict

    return final_dict

    
"""
Legacy date lists to input into run_sim function
"""
DATES_to_SIM = [datetime(2022,11,4), datetime(2022,11,18), datetime(2022,12,2), datetime(2022,12,9), datetime(2022,12,16), datetime(2022,12,23), datetime(2023,1,6),datetime(2023,1,13),datetime(2023,1,20),datetime(2023,1,27),]
FINAL_DATES = [datetime(2024,3,31)]


def run_sim(dates_to_sim, final_dates):
    """
    This is the main function of the entire script
    The function first aggregates input data from bloomberg or other input sources and then calls mc_test_new to run Monte Carlo Simulation
    """
    security_dict = {'SB1 Comdty':'NY No.11',
                    'BAAWHYDP Index':'Hydrous Ethanol',
                    'BAAWANAB Index':'Anhydrous Ethanol',
                    'GCFPUBGC Index':'Fertilizer Costs',
                    'CO1 Comdty':'Brent Crude',
                    'USDBRL Curncy':'USDBRL',
                    'BZCESECA Index':'Energy Prices',
                    'SBMAR1 Comdty':'SBMAR1 Comdty',
                    'SBMAY1 Comdty':'SBMAY1 Comdty',
                    'SBJUL1 Comdty':'SBJUL1 Comdty',
                    'SBOCT1 Comdty':'SBOCT1 Comdty',
                    'SBMAR2 Comdty':'SBMAR2 Comdty'}


    con = pdblp.BCon(debug=False, port=8194, timeout=100000)
    try:
        con.start()
    except:
        print("API Connection Failed")

    ticker_ls = list(security_dict.keys())
    end_date = datetime.today() 
    start_date =  date(end_date.year - 5, end_date.month, end_date.day).strftime("%Y%m%d")
    end_date = end_date.strftime("%Y%m%d")

    data_df = con.bdh(ticker_ls, 'PX_LAST', start_date, end_date) #Grabs data from Bloomberg

    
    column_ls = []
    for col in range(0, data_df.shape[1]):
        temp_col = data_df.columns[col][0]
        column_ls.append(security_dict[temp_col])

    data_df.columns = column_ls
    data_df['Date'] = data_df.index
    final_dict_ls = []
    final_df_aggregate = []
    for tt in range(len(final_dates)):
        for zzz in range(len(dates_to_sim)):
            x = mc_test_new(dates_to_sim[zzz], final_dates[tt], data_df=data_df)
            final_dict_ls.append(x)
            x_df = pd.DataFrame.from_dict(x['sugar_1'])
            for keys in x.keys():
                temp_x_df = pd.DataFrame.from_dict(x[keys])
                if len(final_df_aggregate) == 0:
                    final_df_aggregate = temp_x_df
                else:
                    final_df_aggregate = pd.concat([final_df_aggregate, temp_x_df], ignore_index=True, axis=0)
        final_df_aggregate.to_csv('backup_save_' + str(zzz) + '.csv')
        return final_df_aggregate
    #final_df_aggregate.to_csv("final_output_mc_sim.csv")

def column_reformate():
    """
    legacy code
    """
    col_ls = ['forecast_period','reference','mean_returned','std_returned','sim_date','end_date']
    data_df = pd.read_csv('final_output_mc_sim.csv', index_col=False)
    index_range = np.arange(0,726,len(col_ls))
    prev_index = 0
    final_df_save = []
    for index in index_range:
        if index == 0:
            continue
        else:
            temp_labels = np.arange(prev_index, index, 1).tolist()
            for i in range(0,len(temp_labels)):
                temp_labels[i] = str(temp_labels[i])
            

            temp_df = data_df.filter(temp_labels, axis = 1)
            temp_df.columns = col_ls
            temp_df = temp_df.dropna()
            if index == 6:
                final_df_save = pd.DataFrame(temp_df)
            else:
                final_df_save = pd.concat([final_df_save, temp_df], axis = 0, ignore_index=True)

        prev_index = index

