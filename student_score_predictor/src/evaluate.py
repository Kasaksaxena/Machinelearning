import matplotlib.pyplot as plt
import seaborn as sns

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')  # Perfect line
    plt.xlabel("Actual Final Scores")
    plt.ylabel("Predicted Final Scores")
    plt.title("🎯 Actual vs Predicted Final Scores")
    plt.tight_layout()
    plt.show()