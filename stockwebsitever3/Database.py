import mysql.connector
from datetime import datetime
from datetime import datetime
import pandas as pd 
from sqlalchemy import create_engine





def read_table(table_name,column_name):
    # Establish a connection to the MySQL database
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12341234",
        database="Stockproject"
    )
    # Create a cursor object using the connection
    cursor = conn.cursor()
    
    dates = []
    closes = []

    try:
        cursor.execute(f"SELECT Date, {column_name} FROM {table_name}")
        rows = cursor.fetchall()
        print("SUCCESS")
        for date, close in rows:
            dates.append(date)
            closes.append(close)
        
        return dates, closes

    except mysql.connector.Error as e:
        print(f"Error reading data from table {table_name}: {e}")
    finally:
        cursor.close()
        conn.close()

def get_most_recent_row_df(table_name):

    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12341234",
        database="Stockproject"
    )
    cursor = conn.cursor()

    try:
        
        cursor.execute(f"SELECT MAX(Date) FROM {table_name}")
        result = cursor.fetchone()
        most_recent_date = result[0] if result else None

        if most_recent_date:
            query = f"SELECT * FROM {table_name} WHERE Date = %s"
            df = pd.read_sql(query, conn, params=(most_recent_date,))
            print("SUCCESS")
            return df

    except mysql.connector.Error as e:
        print(f"Error reading data from table {table_name}: {e}")

    finally:

        cursor.close()
        conn.close()

    return pd.DataFrame()



def read_last_20(table_name, specific_date):
    # Convert specific_date from datetime to string in the format YYYY-MM-DD if it's not a string
    if isinstance(specific_date, datetime):
        specific_date = specific_date.strftime('%Y-%m-%d')
    if not isinstance(specific_date, str):
        print(specific_date)
        raise ValueError("specific_date must be a string in 'YYYY-MM-DD' format or a datetime object")

   
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12341234",
        database="Stockproject"
    )
    cursor = conn.cursor()

    # Define the columns to select
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'garman_klass_vol', 'rsi_standardized',
               'bb_low', 'bb_mid', 'bb_high', 'atr', 'macd', 'dollar_volume',
               'text_analysis_number', 'Close_Percent_Change']

    try:
        query = f"SELECT {', '.join(columns)} FROM {table_name} WHERE Date <= %s ORDER BY Date DESC LIMIT 20"
        cursor.execute(query, (specific_date,))
        rows = cursor.fetchall()
        print("SUCCESS")

        # Create a DataFrame from the rows fetched
        df = pd.DataFrame(rows, columns=columns)

        # Reverse the DataFrame to make the dates chronological
        df = df.iloc[::-1].reset_index(drop=True)

        return df

    except mysql.connector.Error as e:
        # Print the error if the query fails
        print(f"Error reading data from table {table_name}: {e}")
    finally:
        # Close the cursor and the connection to free resources
        cursor.close()
        conn.close()




def get_most_recent_date(table_name):
    # Establish a connection to the MySQL database
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="12341234",
        database="Stockproject"
    )
    # Create a cursor object using the connection
    cursor = conn.cursor()

    most_recent_date = None

    try:
        # Execute a SQL query to select the maximum 'Date', which is the most recent date
        cursor.execute(f"SELECT MAX(Date) FROM {table_name}")
        
        # Fetch the result
        result = cursor.fetchone()
        if result:
            most_recent_date = result[0]
            print("SUCCESS")

        return most_recent_date

    except mysql.connector.Error as e:
        # Print the error if the query fails
        print(f"Error reading data from table {table_name}: {e}")
    finally:
        # Close the cursor and the connection to free resources
        cursor.close()
        conn.close()

    return most_recent_date






def insert_dataframe_to_database(df, table_name):
    database_details = {
        'host': 'localhost',
        'user': 'root',
        'password': '12341234',
        'database': 'Stockproject'
    }
    # Extract database connection details
    connection_string = f"mysql+mysqlconnector://{database_details['user']}:{database_details['password']}@{database_details['host']}/{database_details['database']}"
    
    # Establish a connection to the MySQL database
    try:
        engine = create_engine(connection_string)
        # Insert the DataFrame into the MySQL table
        df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
        print("Data inserted successfully into the table.")
    except Exception as e:
        # Print the error if the operation fails
        print(f"Error inserting data into table {table_name}: {e}")
    finally:
        # Dispose the engine to free resources
        engine.dispose()




def fetch_recent_rows_as_dataframe(table_name):

    database_details = {
        'host': 'localhost',
        'user': 'root',
        'password': '12341234',
        'database': 'Stockproject'
    }

    host = database_details['host']
    user = database_details['user']
    password = database_details['password']
    database = database_details['database']
    
    # Create a connection string for SQLAlchemy engine
    connection_string = f"mysql+mysqlconnector://{user}:{password}@{host}/{database}"
    
    # Create the database engine
    engine = create_engine(connection_string)

    try:
        # Assuming there's a column named 'Date' to sort the entries. Adjust the column name as necessary.
        query = f"SELECT * FROM {table_name} ORDER BY Date DESC LIMIT 4"
        
        # Use pandas to execute the query and fetch the data into a DataFrame
        df = pd.read_sql_query(query, con=engine)
        print("DataFrame fetched successfully.")

    except Exception as e:
        # Print the error if the operation fails
        print(f"Error fetching data from table {table_name}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

    finally:
        # Dispose the engine to free resources
        engine.dispose()

    return df


def merge_tables_by_date(table1, table2, column1, column2):
    connection_string = "mysql+mysqlconnector://root:12341234@localhost/Stockproject"
    engine = create_engine(connection_string)
    
    try:
        query1 = f"SELECT Date, `{column1}` AS {column1}_{table1} FROM {table1} ORDER BY Date DESC LIMIT 10"
        query2 = f"SELECT Date, `{column2}` AS {column2}_{table2} FROM {table2} ORDER BY Date DESC LIMIT 10"
        
        df1 = pd.read_sql(query1, engine, parse_dates=['Date'])
        df2 = pd.read_sql(query2, engine, parse_dates=['Date'])

        # Convert datetime to date
        df1['Date'] = pd.to_datetime(df1['Date']).dt.date
        df2['Date'] = pd.to_datetime(df2['Date']).dt.date

        merged_df = pd.merge(df1, df2, on='Date', how='outer')
        
        
        # Round numeric columns to two decimal places
        for column in merged_df.select_dtypes(include=['float64', 'float32']):
            merged_df[column] = merged_df[column].round(4)
        
        merged_df.fillna('DNA', inplace=True)
        merged_df.sort_values(by='Date', ascending=False, inplace=True)

        print("Merge Successful")
        return merged_df

    except Exception as e:
        print(f"Error processing data: {e}")
    
    finally:
        engine.dispose()