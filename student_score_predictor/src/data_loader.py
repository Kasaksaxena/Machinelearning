import pandas as pd 
def load_student_data(file_path):
    df=pd.read_csv(file_path)
    
    print("Data preview:\n ", df.head())
    print("\nData types\n", df.dtypes)
    print("\n Missing values : \n",df.isnull().sum())
    print("\n Shape:\n",df.shape)
    
    return df
    