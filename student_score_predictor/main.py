from src.data_loader import load_student_data
from src.clean import clean_student_data
from src.visualize import visualize_data
from src.model import train_and_evaluate_model


def main():
    df=load_student_data("student_score_predictor\data\students.csv")
    print("\n Cleaning Data : \n")
    
    cleaned_df=clean_student_data(df)
    #print("\n Cleaned data Preview:\n",cleaned_df.head())
    print("\n visualization :\n")
    visualize_data(cleaned_df)
    print("\n🧪 X and y shape check:")
    print(f"X shape: {cleaned_df.drop(columns=['final_score']).shape}")
    print(f"y shape: {cleaned_df['final_score'].shape}")
    print("\n Training model.. :\n")
    model=train_and_evaluate_model(cleaned_df)
    

if __name__=="__main__":
    main()