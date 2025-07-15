import pandas as pd

def preprocess_data(df):
    df=df.drop(columns=['salary_currency','employee_residence'])
    
    caterogical_cols=["experience_level","employment_type","job_title","company_location","company_size"]
    df_encoded=pd.get_dummies(df,columns=caterogical_cols,drop_first=True)
    
    #split
    X=df_encoded.drop('salary',axis=1)
    y=df_encoded["salary"]
    
    #reset index
    X=X.reset_index(drop=True)
    y=y.reset_index(drop=True)
    print(X.info())
    print(y.info())
    return X,y