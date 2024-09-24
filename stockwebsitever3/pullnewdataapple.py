import praw
import pandas as pd
from datetime import datetime

# Initialize PRAW with your Reddit credentials
reddit = praw.Reddit(
    client_id='CQ_7GpsUPyWovw2bAAqwtQ',
    client_secret='RSDca4ioVam2PRQ3PA6D8_pQ_Z0Apg',
    user_agent='SentimentalPractice',
)

def fetch_submissions(subreddit_name, keyword, time_filter, start_date, end_date, limit=500):
    # Convert start_date and end_date from datetime to Unix timestamp
    start_timestamp = datetime.strptime(start_date, '%Y.%m.%d').timestamp()
    end_timestamp = datetime.strptime(end_date, '%Y.%m.%d').timestamp()

    # Search for submissions in the subreddit
    submissions = reddit.subreddit(subreddit_name).search(keyword, time_filter=time_filter, limit=limit)

    # Collect and filter information from each submission, including the content (selftext)
    data = []
    for submission in submissions:
        if start_timestamp <= submission.created_utc <= end_timestamp:
            data.append({
                'title': submission.title,
                'score': submission.score,
                'id': submission.id,
                'url': submission.url,
                'created': datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                'num_comments': submission.num_comments,
                'selftext': submission.selftext  # Capturing the content of the post
            })

    data.sort(key=lambda x: x['created'])

    return pd.DataFrame(data)
from transformers import BertForSequenceClassification, BertTokenizer
import torch

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model_path = 'finetunedtransformer'

model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
def get_model_output(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_number = logits.squeeze().item()
    return predicted_number
from tqdm import tqdm


def get_text_analysis(result_df):
  output_col = []
  for index, row in tqdm(result_df.iterrows(), total=len(result_df)):
      text = row['selftext']
      output = get_model_output(text)
      output_col.append(output)
  result_df['text_analysis_number'] = output_col
  result_df['datetime'] = pd.to_datetime(result_df['created'])
  result_df['Date'] = result_df['datetime'].dt.date
  result_df['Date']=pd.to_datetime(result_df['Date'])
  #df['date_only'] = df['datetime_col'].dt.date
  result_df = result_df.set_index('Date')
  result_df = result_df.drop(['title','score','id','url','num_comments','selftext','datetime','created'],axis=1)
  result_df = result_df.groupby('Date').mean();
  return result_df

import yfinance as yf
import pandas_ta as ta
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def compute_indicators(stock_data):
    # Compute ATR
    atr = ta.atr(high=stock_data['High'],
                        low=stock_data['Low'],
                        close=stock_data['Close'],
                        length=14)
    atr_normalized = (atr - atr.mean()) / atr.std()


    # Compute MACD
    macd = ta.macd(close=stock_data['Close'], length=20) .iloc[:, 0]
    macd_normalized = (macd - macd.mean()) / macd.std()

    # Compute dollar volume
    dollar_volume = (stock_data['Adj Close'] * stock_data['Volume']) / 1e6

    return atr_normalized, macd_normalized, dollar_volume

def get_reddit(startdate,enddate,compname):
    subreddit_name = 'stocks'
    keyword = compname
    time_filter = 'year'
    start_date = startdate.strftime("%Y.%m.%d")
    end_date = enddate.strftime("%Y.%m.%d")
    limit = 500
    df = fetch_submissions(subreddit_name, keyword, time_filter, start_date, end_date, limit)
    if(len(df)==0):
      return df

    df=get_text_analysis(df)
    return df



def getstocknumber(startdate, enddate,ticker,compname):
   
    df2 = get_reddit(startdate,enddate,compname)

    # Modify the start date to be 30 days earlier
    startdate = startdate - timedelta(days=60)
    startdate_str = startdate.strftime("%Y-%m-%d")



    # Download stock data
    df1 = yf.download(ticker, start=startdate, end=enddate)

    # ------------


    df1['Close_Change'] = df1['Close'].diff()
    df1['Close_Percent_Change'] = df1['Close'].pct_change() * 100
    #Last day percent change add
    df1['Last_day_percent_change'] = df1['Close_Percent_Change'].shift(+1)
    df1 = df1.iloc[1:]
    #result_df = pd.merge(df1, df2, left_index=True, right_index=True)

    result_df = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')


    result_df.dropna(subset=['Close'], inplace=True)

    result_df['garman_klass_vol'] = ((np.log(result_df['High']) - np.log(result_df['Low'])) ** 2) / 2 - (2 * np.log(2) - 1) * ((np.log(result_df['Adj Close']) - np.log(result_df['Open'])) ** 2)
    result_df['rsi'] = ta.rsi(close=result_df['Adj Close'], length=14)
    result_df['rsi_standardized'] = (result_df['rsi'] - result_df['rsi'].mean()) / result_df['rsi'].std()
    bbands_df = ta.bbands(close=np.log1p(result_df['Adj Close']), length=20)

    result_df['bb_low'] = bbands_df.iloc[:, 0]
    result_df['bb_mid'] = bbands_df.iloc[:, 1]
    result_df['bb_high'] = bbands_df.iloc[:, 2]

    atr, macd, dollar_volume = compute_indicators(result_df)
    result_df['atr'] = atr
    result_df['macd'] = macd
    result_df['dollar_volume'] = dollar_volume


    return result_df.tail(1)
