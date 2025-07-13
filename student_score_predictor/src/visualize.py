import seaborn as sns 
import matplotlib.pyplot as plt 

def visualize_data(df):
    #set a style
    sns.set_theme(style="whitegrid",palette="pastel")
    
    #Histogram of final scores
    plt.figure(figsize=(8,5))
    sns.histplot(df["final_score"],bins=10,kde=True)
    plt.title("Final Exam Score Distribution")
    plt.xlabel("Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.show()
    
    
    #Scatter : study hours vs  final score
    plt.figure(figsize=(8,5))
    sns.scatterplot(x="study_hours",y="final_score",data=df)
    plt.title("Study Hours vs Final Score")
    plt.xlabel("Study Hours")
    plt.ylabel("Final Score")
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True),annot=True,cmap="YlGnBu",fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.show()
    
    
    