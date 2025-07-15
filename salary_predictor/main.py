from src.loader import load_data
from src.preprocess import preprocess_data


if __name__=="__main__":
    df=load_data()
    X,y=preprocess_data(df)
    
    print("\n Processing complete \n")
    print("features shape",X.shape)
    print("Target shape",y.shape)
    