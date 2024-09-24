#import pullnewdataapple
import Database
import pullnewdataapple
import FinetuneLSTM

import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import praw
import pandas as pd
from transformers import BertForSequenceClassification, BertTokenizer
import torch
from tqdm import tqdm
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import datetime




def checktommorowdata(startdate,Databaseto,ticker,compname):
    weekday = startdate.weekday()
    if weekday >= 5:  # Saturday (5) or Sunday (6)
        print("The date falls on a weekend, skipping data fetch.")
        return  # Exit the function
    enddate = startdate + timedelta(days=1)
    print("Fetching data for date:", startdate)
    
    try:
        temp = yf.download(ticker, start=startdate, end=enddate)
        if not temp.empty:
            most_recent_data_date = temp.index[-1].date()
            if(most_recent_data_date != startdate):
                print("Same day")
                return
            else:
                new_data = pullnewdataapple.getstocknumber(startdate, enddate,ticker,compname)
                if 'text_analysis_number' in new_data.columns:
                    new_data['text_analysis_number'].fillna(0, inplace=True)
                else:
                    print("Warning: 'text_analysis_number' column not found.")
                    new_data['text_analysis_number'] = 0  # or handle appropriately
                new_data.reset_index(inplace=True)
                new_data.rename(columns={'index': 'Date'}, inplace=True)
                Database.insert_dataframe_to_database(new_data, Databaseto)
                
        else:
            print("No data retrieved from yfinance.")

    except Exception as e:
        print(f"Yfinance is empty on the date: {e}")
        

def predictnext(arr,date,Databasefrom,compname):
    temp2= Database.read_last_20(Databasefrom,date)
    temp =predicttommorow(temp2,compname)
    temp = temp.reset_index()
    temp['Date'] = pd.to_datetime(temp['Date'], errors='coerce')
    arr = pd.concat([arr,temp.tail(1)])
    return arr

def create_input_sequences(data, n_steps):
        X = []
        for i in range(len(data) - n_steps):
            X.append(data[i:(i + n_steps), :])  # Create sliding windows of size n_steps
        return np.array(X)



def predicttommorow(new_data,compname):
    with open(f'LSTM/{compname}/scaler.pkl', 'rb') as file:
        scaler_model1 = pickle.load(file)
    expected_features = ['Open', 'High', 'Low', 'Close', 'garman_klass_vol', 'rsi_standardized',
                        'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd', 'dollar_volume',
                        'text_analysis_number', 'Close_Percent_Change']
    new_data_for_scaling = new_data[expected_features]
    n_steps = 10
    new_data_scaled_model1 = scaler_model1.transform(new_data_for_scaling)
    X_new_model1 = create_input_sequences(new_data_scaled_model1, n_steps)
    model1 = load_model(f'LSTM/{compname}/modelsave')
    predictions1 = model1.predict(X_new_model1)
    predictions1_actual = predictions1.reshape(-1, 1)
    # index_of_close = expected_features.index('Close')
    # close_min1 = scaler_model1.min_[index_of_close]
    # close_scale1 = scaler_model1.scale_[index_of_close]
    # predictions1_actual = (predictions1_reshaped - close_min1) / close_scale1
    
    if 'Date' in new_data.columns:
        new_data['Date'] = pd.to_datetime(new_data['Date'])  # Convert 'Date' column to datetime
        new_data.set_index('Date', inplace=True)  # Set 'Date' as the index
    else:
        raise ValueError("Date column missing from data")

    prediction_dates = new_data.index[n_steps:] + pd.Timedelta(days=1)
    predictions_df = pd.DataFrame(predictions1_actual, index=prediction_dates, columns=['Predicted_Close'])
    return predictions_df
        
def has_stock_data_on_date(symbol, specific_date):
    specific_date_dt = pd.to_datetime(specific_date)
    end_date_dt = specific_date_dt + timedelta(days=1)
    end_date_str = end_date_dt.strftime('%Y-%m-%d')
    stock_data = yf.download(symbol, start=specific_date, end=end_date_str)
    return not stock_data.empty




def updatetodatabase(day,ticker, Databasefrom,Databaseto,compname):
    predicted_df = pd.DataFrame(columns=['Date', 'Predicted_Close'])
    predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
    predicted_df['Predicted_Close'] = predicted_df['Predicted_Close'].astype(float)
    date = Database.get_most_recent_date(Databasefrom).date()
    #date = datetime.date.today()

    for i in range(day):
        
        date+= timedelta(days=1)
        datestr = date.strftime('%Y-%m-%d')
       
        
        if(has_stock_data_on_date(ticker,datestr)): 
            checktommorowdata(date,Databasefrom,ticker,compname)        
            predicted_df=predictnext(predicted_df,datestr,Databasefrom,compname)        
            Database.insert_dataframe_to_database(predicted_df.tail(1), Databaseto)
            FinetuneLSTM.finetunemodel(Databasefrom,compname)

        else:
            print("Dont have date today!")
        
    zz = Database.fetch_recent_rows_as_dataframe(Databasefrom)



#updatetodatabase(120,"MSFT","Microsoftstock","Microsoftpredicted","Microsoft")
#updatetodatabase(120,"AAPL","Applestock","Applepredicted","Apple")
#updatetodatabase(100,"TSLA","Teslastock","Teslapredicted","Tesla")
#updatetodatabase(100,"NVDA","Nvidiastock","Nvidiapredicted","Nvidia")
#updatetodatabase(40,"AAPL","Applestock","Applepredicted","Apple")
updatetodatabase(40,"AAPL","Applestock","Applepredicted","Apple")
