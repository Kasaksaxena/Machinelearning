from src.loader import load_data
from src.preprocess import preprocess_data
from src.odel import train_linear_model, train_polynomial_model,train_lasso_model,train_ridge_model
from sklearn.model_selection import train_test_split
from src.evaluate import evaluate_model
if __name__=="__main__":
    df=load_data()
    X,y=preprocess_data(df)
    
    #Split
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
   
    print("\n Training Linear Regression \n")
    linear_model=train_linear_model(X_train,y_train)
    
    print("\n Training Polynomial Regression (degree=2) \n")
    poly_model=train_polynomial_model(X_train,y_train,degree=2 )
    
    print("\n Training ridge  \n")
    ridge_model=train_ridge_model(X_train,y_train,alpha=1.0)
    
    print("\n Training Lasso  \n")
    lasso_model=train_lasso_model(X_train,y_train,alpha=0.1)
   
    
    print("\n Model Trained Successfully!")
    
    # evaluate
    evaluate_model(linear_model,X_test,y_test,title="Linear Regression")
    evaluate_model(poly_model,X_test,y_test,title="Polynomial Regression(Degree 2)")
    evaluate_model(ridge_model,X_test,y_test,title="Ridge Regression")
    evaluate_model(lasso_model,X_test,y_test,title="Lasso Regression")
    
    
    
    # print("\n Processing complete \n")
    # print("features shape",X.shape)
    # print("Target shape",y.shape)
    