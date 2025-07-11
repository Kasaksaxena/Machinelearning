import matplotlib.pyplot as plt 

def plot_predictions(y_test,y_pred):
    plt.figure(figsize=(10,6))
    plt.scatter(y_test,y_pred,alpha=0.5,color="royalblue")
    plt.xlabel("ACTUAL HOME PRICES")
    plt.ylabel("PREDICTED HOUSE PRICES")
    plt.title("ACTUAL VS  PREDICTED PRICES")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("prediction_plot.png")
    plt.show()