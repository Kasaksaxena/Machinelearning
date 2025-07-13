from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score 
from src.evaluate import plot_predictions
def train_and_evaluate_model(df):
    X=df.drop(columns=["final_score"])
    y=df["final_score"]
    print(f"\n✅ Shapes - X: {X.shape}, y: {y.shape}")
    #split data 80%  training, 20 % testing 
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
    
    #Create model and train
    model=LinearRegression()
    model.fit(X_train,y_train)
    
    # predict on test data
    y_pred=model.predict(X_test)
    
    #evaluate
    mse=mean_squared_error(y_test,y_pred)
    r2=r2_score(y_test,y_pred)
    
    print("\n Model Evaluation")
    print(f"Mean Squared  Error {mse:.2f}")
    print(f"R2 Score {r2:.2f}")
    
    plot_predictions(y_test,y_pred)
    
    return model