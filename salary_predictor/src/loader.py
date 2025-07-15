import pandas as pd

def load_data(filepath='salary_predictor/data/salary.csv'):
    df=pd.read_csv(filepath)
    print("Data loaded Successfully")
    print("First 5 Rows")
    print(df.head())
    print("\n info \n")
    print(df.info())
    print("\n shape of data",df.shape)
    return df