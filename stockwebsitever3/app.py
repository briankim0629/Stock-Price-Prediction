from flask import Flask, render_template
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import Database
import datetime
app = Flask(__name__)

def simplifyforplot(ap_date,ap_close,howmany):
    data = pd.DataFrame({
        'Date': ap_date,
        'Close': ap_close
    })

    # Convert 'Date' from string to datetime object, necessary for correct sorting and plotting
    data['Date'] = pd.to_datetime(data['Date'])

    # Sort the DataFrame by 'Date' in ascending order
    data = data.sort_values(by='Date', ascending=True)

    # Select only the last 100 entries for the most recent data
    data = data.tail(howmany)

    # Extract the 'Date' and 'Close' columns for plotting
    ap_date = data['Date']
    ap_close = data['Close']
    return ap_date,ap_close

def get_connected_dots(Databasein,Databasepred):
    # Retrieve the most recent rows from the respective tables
    temp1 = Database.get_most_recent_row_df(Databasein)
    temp2 = Database.get_most_recent_row_df(Databasepred)

    # Before merging, select only the required columns in each DataFrame
    # Ensure temp1 already contains 'Close'
    temp1 = temp1[['Date', 'Close']]
    # Map 'Predicted_Close' to 'Close' in temp2, then select only the 'Date' and the renamed 'Close'
    temp2 = temp2.rename(columns={'Predicted_Close': 'Close'})
    temp2 = temp2[['Date', 'Close']]

    # Merge the two dataframes on 'Date'
    #merged_df = pd.merge(temp1, temp2, on='Date', suffixes=('', '_Predicted'))
    merged_df = pd.concat([temp1,temp2],axis = 0)

    return merged_df


def generate_plots():
    ap_date, ap_close = Database.read_table("Applestock","Close")
    predap_date, predap_close = Database.read_table("Applepredicted","Predicted_Close")
   
    ap_date, ap_close = simplifyforplot(ap_date,ap_close,60)
    predap_date, predap_close = simplifyforplot(predap_date,predap_close,60)
    connected_df = get_connected_dots("Applestock","Applepredicted")
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=ap_date, y=ap_close, mode="lines", name="Apple Close"))
    fig1.add_trace(go.Scatter(x=predap_date, y=predap_close, mode="lines", name="Predicted_change")) #, line=dict(dash='dot')))
    fig1.add_trace(go.Scatter(x=connected_df["Date"], y=connected_df["Close"], mode="lines", name="Predicted_Movement"))
    fig1.update_layout(
        title="Apple Stock Closing Prices",
        xaxis_title="Date",
        yaxis_title="Close Price",
        xaxis_tickangle=-45,
    )
    #---------------------------------------------------------------------------
    # Create the second line plot for Microsoft
    tsla_date, tsla_close = Database.read_table("Teslastock","Close")
    predtsla_date, predtsla_close = Database.read_table("Teslapredicted","Predicted_Close")
   
    tsla_date, tsla_close = simplifyforplot(tsla_date,tsla_close,100)
    predtsla_date, predtsla_close = simplifyforplot(predtsla_date,predtsla_close,100)
    connected_df3 = get_connected_dots("Teslastock","Teslapredicted")
    fig3 = go.Figure()
    #fig2.add_trace(go.Scatter(x=microsoft_data["Date"], y=microsoft_data["Close"], mode="lines", name="Microsoft Close"))
    fig3.add_trace(go.Scatter(x=tsla_date, y=tsla_close, mode="lines", name="Tesla Close"))
    fig3.add_trace(go.Scatter(x=predtsla_date, y=predtsla_close, mode="lines", name="Predicted_change")) #, line=dict(dash='dot')))
    fig3.add_trace(go.Scatter(x=connected_df3["Date"], y=connected_df3["Close"], mode="lines", name="Predicted_Movement"))
    fig3.update_layout(
        title="Tesla Stock Closing Prices",
        xaxis_title="Date",
        yaxis_title="Close Price",
        xaxis_tickangle=-45,
    )


    # Convert to JSON for rendering
    return pio.to_json(fig1), pio.to_json(fig3)

def get_table():
    plotApple, plotTesla = generate_plots()
    today = datetime.date.today()
    today_str = today.strftime('%Y-%m-%d')
    #data2=Database.read_last_20("Applestock",today_str)
    data = Database.merge_tables_by_date("Applestock","Applepredicted","Close","Predicted_Close")
    data2 = Database.merge_tables_by_date("Teslastock","Teslapredicted","Close","Predicted_Close")
    AppleData1 = ["Date"] + data.iloc[:9, 0].tolist()[::-1]
    AppleData2 = ["Actual"] + data.iloc[:9, 1].tolist()[::-1]
    AppleData3 = ["Predicted"] + data.iloc[:9, 2].tolist()[::-1]
    TeslaData1 = ["Date"] + data2.iloc[:9, 0].tolist()[::-1]
    TeslaData2 = ["Actual"] + data2.iloc[:9, 1].tolist()[::-1]
    TeslaData3 = ["Predicted"] + data2.iloc[:9, 2].tolist()[::-1]
  
    
    # tableData1 = data.iloc[:9, 0].tolist() 
    # tableData2 = data.iloc[:9, 1].tolist()  
    # tableData3 = data.iloc[:9, 2].tolist()
    
    return AppleData1,AppleData2,AppleData3, TeslaData1, TeslaData2, TeslaData3





@app.route("/")
def home():
    
    plotApple, plotTesla = generate_plots()
    AppleData1, AppleData2,AppleData3, TeslaData1, TeslaData2,TeslaData3 = get_table()
    RecentDateApple = AppleData1[-2]
    RecentDateTesla = TeslaData1[-2]
    return render_template("index.html", plotApple=plotApple,  plotTesla = plotTesla, AppleData1=AppleData1, AppleData2=AppleData2, AppleData3 = AppleData3, TeslaData1 = TeslaData1, TeslaData2 = TeslaData2, TeslaData3= TeslaData3, RecentDateApple = RecentDateApple, RecentDateTesla = RecentDateTesla)

if __name__ == "__main__":
    app.run(debug=True)

