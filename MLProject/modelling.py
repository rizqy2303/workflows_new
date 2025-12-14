import pandas as pd
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
import mlflow
import mlflow.sklearn
import warnings
import dagshub 
import os 

warnings.filterwarnings("ignore")

def preprocess_data(df):
    """Melakukan preprocessing sederhana: drop kolom, one-hot encode, dan fillna 0."""
    cols_to_drop = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    df = df.fillna(0)
    return df

def main():
    
    dagshub.init(repo_owner='rizqyputrap23', repo_name='my-first-repo', mlflow=True) 
    
    mlflow.set_experiment("Eksperimen_Advance_DagsHub_Final")
    
    file_train = "train.csv" 
    file_test = "test.csv" 
    TARGET_COLUMN = "Survived" 

    try:
        train_df = pd.read_csv(file_train) 
        test_df = pd.read_csv(file_test)
    except FileNotFoundError:
        print("ERROR: Pastikan file 'train.csv' dan 'test.csv' ada.")
        return 
    combined_df = pd.concat([train_df.drop(columns=[TARGET_COLUMN], errors='ignore'), test_df], ignore_index=True)
    combined_processed = preprocess_data(combined_df)

    X_train_processed = combined_processed.iloc[:len(train_df)].drop(columns=[TARGET_COLUMN], errors='ignore')
    X_test_processed = combined_processed.iloc[len(train_df):].drop(columns=[TARGET_COLUMN], errors='ignore')
    y_train = train_df[TARGET_COLUMN]
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="Best_LogReg_Autolog"):
        
        param_grid = {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}
        logreg = LogisticRegression(random_state=42, solver='liblinear') 
        grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', verbose=0)
        
        print("Fitting Grid Search (Autolog aktif)...")
        grid_search.fit(X_train_processed, y_train)
        
        print(f"\n--- Logging Otomatis Berhasil ---")
        print(f"Best Parameters: {grid_search.best_params_}")
        print("Cek DagsHub/MLflow UI untuk melihat Artifacts (model, conda.yaml, dll).")

if __name__ == "__main__":
    main()
# Script berhasil dijalankan
print("CI script success")