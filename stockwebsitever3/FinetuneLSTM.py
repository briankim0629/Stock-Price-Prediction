# import datetime
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import pandas as pd

# import Database  # Your custom module

# # Get today's date and format it
# today = datetime.date.today()
# today_str = today.strftime('%Y-%m-%d')

# # Load the pre-trained LSTM model
# model = load_model('LSTM/modelsave')

# # Fetch the last 20 days of data for Microsoft stock as of today
# data_window = Database.read_last_20("Microsoftstock", today_str)
# print("Data Window:")
# print(data_window)

# # Validate the data and convert 'date' column to index if present
# if 'Date' in data_window.columns:
#     data_window.set_index('Date', inplace=True)

# # Ensure all data is in float32 for TensorFlow compatibility
# data_window = data_window.astype(np.float32)

# # Check for any missing data
# if data_window.isnull().any().any():
#     raise ValueError("Data contains NaNs")

# # Convert DataFrame to numpy array
# data_array = data_window.to_numpy()

# # Validate that there are at least 10 days of data
# if len(data_array) < 10:
#     raise ValueError("Not enough data to form input")

# # Use the last 10 days of data for the model's input
# data_array = data_array[-10:]

# # Reshape data for the model
# input_data = data_array.reshape(1, 10, 14)
# print("Input Data Shape:")
# print(input_data.shape)
# print(input_data)

# # Assume the last day's full feature set as the target
# target_data = data_array[-1].reshape(1, 14)
# print("Target Data Shape:")
# print(target_data.shape)
# print(target_data)

# # Fine-tune the model
# model.fit(input_data, target_data, epochs=1)

# # Save the fine-tuned model with a date-stamped filename
# model_name = f'LSTM/microsoftmodelsaved/modelsaved_{today_str}.h5'
# model.save(model_name)

# # Optional: Make a prediction
# prediction = model.predict(input_data)
# print("Prediction for the next day's feature set:")
# print(prediction)


import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import Database
import datetime
import decimal



import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import MinMaxScaler


# today = datetime.date.today()
# today_str = today.strftime('%Y-%m-%d')

# # Fetch new data from the database
# new_data = Database.read_last_20("Microsoftstock", today_str)
# # print(new_data)

# with open('LSTM/scaler.pkl', 'rb') as file:
#     scaler_model1 = pickle.load(file)

# # print("Features expected by Model 1's scaler:", scaler_model1.feature_names_in_)
# expected_features = ['Date','Open', 'High', 'Low', 'Close', 'garman_klass_vol', 'rsi_standardized',
#                      'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd', 'dollar_volume',
#                      'text_analysis_number', 'Close_Percent_Change']
# new_data_for_scaling = new_data[expected_features]
# # Print current features in new data
# # print("Current features in new data:", new_data.columns.tolist())

# new_data_for_scaling['Close_Next_Day'] = new_data_for_scaling['Close'].shift(-1)
# # print(new_data_for_scaling)
# new_data_for_scaling = new_data_for_scaling.dropna()
# # print("NEW")
# # print(new_data_for_scaling)
# print("Training for date : " , new_data_for_scaling.iloc[-1]["Date"])

# #new_data_scaled_model1 = scaler_model1.transform(new_data_for_scaling)
# expected_features = ['Open', 'High', 'Low', 'Close', 'garman_klass_vol', 'rsi_standardized',
#                      'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd', 'dollar_volume',
#                      'text_analysis_number', 'Close_Percent_Change']
# features_to_scale = new_data_for_scaling[expected_features]
# scaled_features = scaler_model1.transform(features_to_scale)
# # Replace the original columns with the scaled ones
# new_data_for_scaling[expected_features] = scaled_features

def create_sequences(data, features, n_steps):
    X, y = [], []
    for i in range(n_steps, len(data)):
        X.append(data[features].iloc[i-n_steps:i].values)
        y.append(data['Close_Next_Day'].iloc[i])
    return np.array(X), np.array(y)

def convert_decimals(data_frame):
    # This function converts all decimal.Decimal types in the DataFrame to floats
    for column in data_frame.columns:
        if data_frame[column].dtype == object:
            # Apply conversion if the column type is object, which may contain decimal.Decimal
            data_frame[column] = data_frame[column].apply(lambda x: float(x) if isinstance(x, decimal.Decimal) else x)


def finetunemodel(Databasename,compname):
    today = datetime.date.today()
    today_str = today.strftime('%Y-%m-%d')

    # Fetch new data from the database
    new_data = Database.read_last_20(Databasename, today_str)
    convert_decimals(new_data)
    # print(new_data)

    with open(f'LSTM/{compname}/scaler.pkl', 'rb') as file:
        scaler_model1 = pickle.load(file)

    # print("Features expected by Model 1's scaler:", scaler_model1.feature_names_in_)
    expected_features = ['Date','Open', 'High', 'Low', 'Close', 'garman_klass_vol', 'rsi_standardized',
                        'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd', 'dollar_volume',
                        'text_analysis_number', 'Close_Percent_Change']
    new_data_for_scaling = new_data[expected_features]
    # Print current features in new data
    # print("Current features in new data:", new_data.columns.tolist())

    new_data_for_scaling['Close_Next_Day'] = new_data_for_scaling['Close'].shift(-1)
    # print(new_data_for_scaling)
    new_data_for_scaling = new_data_for_scaling.dropna()
    # print("NEW")
    # print(new_data_for_scaling)

    #new_data_scaled_model1 = scaler_model1.transform(new_data_for_scaling)
    expected_features = ['Open', 'High', 'Low', 'Close', 'garman_klass_vol', 'rsi_standardized',
                        'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd', 'dollar_volume',
                        'text_analysis_number', 'Close_Percent_Change']
    features_to_scale = new_data_for_scaling[expected_features]
    scaled_features = scaler_model1.transform(features_to_scale)
    # Replace the original columns with the scaled ones
    new_data_for_scaling[expected_features] = scaled_features
    features = ['Open', 'High', 'Low', 'Close', 'garman_klass_vol', 'rsi_standardized', 'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd', 'dollar_volume', 'text_analysis_number', 'Close_Percent_Change']
    n_steps = 10
    X, y = create_sequences(new_data_for_scaling.dropna(), features ,n_steps)  # Drop NA to avoid issues in training

    model = load_model(f'LSTM/{compname}/modelsave')
    model.fit(X, y, epochs=1, batch_size=1) 
    model.save(f'LSTM/{compname}/modelsave')
    print("Training for date : " , new_data_for_scaling.iloc[-1]["Date"])
    print("Fine tuned completed ")


# features = ['Open', 'High', 'Low', 'Close', 'garman_klass_vol', 'rsi_standardized', 'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd', 'dollar_volume', 'text_analysis_number', 'Close_Percent_Change']
# n_steps = 10
# X, y = create_sequences(new_data_for_scaling.dropna(), features ,n_steps)  # Drop NA to avoid issues in training
#print(X)



#new_data_scaled_model1 = scaler_model1.transform(new_data_for_scaling)
