from src.data_loader import load_student_data
from src.clean import clean_student_data
from src.visualize import visualize_data
def main():
    df=load_student_data("student_score_predictor\data\students.csv")
    print("\n Cleaning Data : \n")
    
    cleaned_df=clean_student_data(df)
    #print("\n Cleaned data Preview:\n",cleaned_df.head())
    print("\n visualization :\n")
    visualize_data(cleaned_df)
if __name__=="__main__":
    main()