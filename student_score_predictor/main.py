from src.data_loader import load_student_data

def main():
    df=load_student_data("student_score_predictor\data\students.csv")
    
if __name__=="__main__":
    main()