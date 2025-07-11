from src.data_loader import load_data
from src.evaluate import evaluate_model
from src.model import train_model
from src.visualize import plot_predictions
from sklearn.model_selection import train_test_split
def main():
    #load data
    df=load_data()
    
    #features and target
    X=df.drop('Target',axis=1)
    y=df["Target"]
    
    #split data
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    
    #train model
    model=train_model(X_train,y_train)
    
    #predict 
    y_pred=model.predict(X_test)
    
    #Evaluate 
    mse,r2=evaluate_model(y_test,y_pred)
    print(f"Mean squared error :{mse:.2f}")
    print(f"R2 score :{r2:2f}")
    
    #plot
    plot_predictions(y_test,y_pred)
    
    
    
if __name__=="__main__":
    main()