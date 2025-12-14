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
    # Hanya drop kolom jika kolom tersebut ada di DataFrame (Penting untuk train/test)
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')
    
    # Tambahkan penanganan data yang lebih baik untuk menghindari error di CI
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)
    df = df.fillna(0) # Mengisi NaN lain yang mungkin tersisa
    return df

def main():
    
    # 1. Inisialisasi DagsHub/MLflow
    # Pastikan variabel lingkungan (secrets) untuk DagsHub sudah diset jika dijalankan di CI
    try:
        dagshub.init(repo_owner='rizqyputrap23', repo_name='my-first-repo', mlflow=True) 
    except Exception as e:
        print(f"Peringatan: Gagal menginisialisasi DagsHub. Model akan dilog ke MLflow lokal. Error: {e}")
    
    mlflow.set_experiment("Eksperimen_Advance_DagsHub_Final")
    
    file_train = "train.csv" 
    file_test = "test.csv" 
    TARGET_COLUMN = "Survived" 

    # 2. Pembacaan Data
    # Di lingkungan CI, file ini harus ada di root atau path yang bisa diakses
    try:
        # Jika Anda menempatkan file di root folder proyek:
        train_df = pd.read_csv(file_train) 
        # Untuk test.csv, asumsikan file ini berada di lokasi yang sama.
        # Catatan: File test.csv biasanya tidak memiliki kolom TARGET, jadi harus di-handle
        test_df = pd.read_csv(file_test)
    except FileNotFoundError:
        print("ERROR: Pastikan file 'train.csv' dan 'test.csv' ada di root directory proyek Anda.")
        # Jika file tidak ada, CI akan gagal di sini.
        return 
    
    # 3. Preprocessing Data
    combined_df = pd.concat([train_df.drop(columns=[TARGET_COLUMN], errors='ignore'), test_df], ignore_index=True)
    combined_processed = preprocess_data(combined_df)
    
    # Menyamakan kolom untuk memecah kembali data (Penting!)
    train_cols = [col for col in combined_processed.columns if col in train_df.columns or col not in ['Survived']]
    
    # Pastikan data yang digunakan untuk training dan testing memiliki kolom yang sama
    X_train_processed = combined_processed.iloc[:len(train_df)].drop(columns=[TARGET_COLUMN], errors='ignore')
    X_test_processed = combined_processed.iloc[len(train_df):]
    
    # Menyelaraskan kolom (misalnya jika ada kolom dummy yang hilang di salah satu dataset)
    # Ini adalah langkah kritis dalam ML
    missing_cols_in_train = set(X_test_processed.columns) - set(X_train_processed.columns)
    for c in missing_cols_in_train:
        X_train_processed[c] = 0

    X_test_processed = X_test_processed[X_train_processed.columns] # Urutkan kolom test agar sama dengan train
    
    y_train = train_df[TARGET_COLUMN]

    # 4. Training dan Logging
    mlflow.sklearn.autolog()
    with mlflow.start_run(run_name="Best_LogReg_Autolog"):
        
        param_grid = {'C': [0.1, 1.0, 10.0], 'penalty': ['l1', 'l2']}
        logreg = LogisticRegression(random_state=42, solver='liblinear') 
        grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', verbose=0)
        
        print("Fitting Grid Search (Autolog aktif)...")
        grid_search.fit(X_train_processed, y_train)
        
        # Log manual jika diperlukan (Autolog sudah mencatat sebagian besar hasil)
        mlflow.log_params(grid_search.best_params_)
        
        print(f"\n--- Logging Otomatis Berhasil ---")
        print(f"Best Parameters: {grid_search.best_params_}")
        print("Cek DagsHub/MLflow UI untuk melihat Artifacts (model, conda.yaml, dll).")

if __name__ == "__main__":
    main()