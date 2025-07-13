import pandas as pd 

def clean_student_data(df):
    # Drop rows where "study_hours" or attendance are missing 
    df = df.dropna(subset=['study_hours','attendance'])
    
    #fill missing "internet usage" with most frequent (mode)
    df["internet_usage"].fillna(df["internet_usage"].mode()[0])
    
    #cap outliers in study_hours to a max of 12
    df["study_hours"]=df["study_hours"].apply(lambda x:min(x,12))
    
    #fill any other missing values with mean(for numeric)
    df=df.fillna(df.mean(numeric_only=True))
    
    #reset index
    df = df.reset_index(drop=True)
    
    # one hot encode "internet usage"
    df=pd.get_dummies(df,columns=["internet_usage"],drop_first=True)
    
    return df