from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import matplotlib.pyplot as plt

def evaluate_model(model,X_test,y_test,title="Model"):
    
    y_predict=model.predict(X_test)
    
    #metrics
    #mse=mean_squared_error(y_test,y_predict)
    mse=mean_absolute_error(y_test,y_predict)
    r2=r2_score(y_test,y_predict)
    print(f"\n {title} Evaluation")
    print(f"Mean squared Error :{ mse:.2f}")
    print(f"R2 Score :{ r2:.2f}")
    
    
    #visualization 
    plt.figure(figsize=(8,5))
    plt.scatter(y_test,y_predict,alpha=0.7,color="blue")
    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title(f"{title}: Actual Vs Predicted")
    plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],"r--")
    plt.grid(True)
    plt.tight_layout()
    plt.show()