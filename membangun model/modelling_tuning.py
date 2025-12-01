# modelling_tuning.py (VERSI PERBAIKAN V2: Mengatasi Error Experiment & Dataset Logging)

import pandas as pd
import os
import sys
from pathlib import Path

# --- PERBAIKAN: Set encoding output konsol ke UTF-8 untuk menghindari error emoji/non-ASCII ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
# ------------------------------------------------------------------------------------------

# Import pustaka Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Import MLflow
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from mlflow.exceptions import MlflowException # Import untuk menangani error


# Path data yang sudah dipreproses
PREPROCESSED_DATA_PATH = Path('amazon_preprocessing') / 'amazon_preprocessed.csv'

# Set nama model yang akan didaftarkan di MLflow
REGISTERED_MODEL_NAME = "Sentiment_LR_Tuned"


def load_data(path: Path):
    """Memuat data training dan testing dari file preprocessed."""
    if not path.exists():
        raise FileNotFoundError(f"Data preprocessed tidak ditemukan di: {path.resolve()}")

    df = pd.read_csv(path)
    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']
    
    X_train, y_train = df_train['review'].astype(str), df_train['sentiment']
    X_test, y_test = df_test['review'].astype(str), df_test['sentiment']
    
    print(f"Data Loaded: Train Samples={len(X_train)}, Test Samples={len(X_test)}")
    return X_train, X_test, y_train, y_test

def create_model_pipeline():
    """Mendefinisikan pipeline model dan ruang hyperparameter untuk tuning."""
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(random_state=42, solver='liblinear'))
    ])
    
    param_distributions = {
        'tfidf__ngram_range': [(1, 1), (1, 2)], 
        'tfidf__max_df': [0.7, 0.85, 1.0], 
        'tfidf__min_df': [1, 5, 10], 
        
        'clf__C': [0.1, 1.0, 10.0], 
        'clf__penalty': ['l1', 'l2'] 
    }
    
    search = RandomizedSearchCV(
        pipeline,
        param_distributions,
        n_iter=10, 
        cv=5,      
        scoring='f1', 
        random_state=42,
        n_jobs=-1, 
        verbose=2
    )
    return search

def evaluate_and_log(search_result, X_test, y_test):
    """Mengevaluasi model terbaik dan melakukan manual logging ke MLflow."""
    best_estimator = search_result.best_estimator_
    y_pred = best_estimator.predict(X_test)
    
    # Hitung Metrik
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_precision": precision_score(y_test, y_pred, zero_division=0),
        "test_recall": recall_score(y_test, y_pred, zero_division=0),
        "test_f1_score": f1_score(y_test, y_pred, zero_division=0),
        "best_cv_score": search_result.best_score_
    }
    
    print("\n--- Model Evaluation (Test Set) ---")
    print(f"F1-Score: {metrics['test_f1_score']:.4f}")

    # --- MANUAL MLFLOW LOGGING ---
    with mlflow.start_run() as run:
        # 1. Log Parameters (Best Params dari Tuning)
        mlflow.log_params(search_result.best_params_)
        
        # 2. Log Metrics
        mlflow.log_metrics(metrics)
        
        # 3. Log Artifacts
        
        # LOGGING DATA PREPROCESSED SEBAGAI ARTEFAK DAN INPUT
        if os.path.exists(PREPROCESSED_DATA_PATH):
            # Log file sebagai Artifact agar file tersedia di UI
            mlflow.log_artifact(str(PREPROCESSED_DATA_PATH), artifact_path="data")
            
            # LOGGING DATA SEBAGAI INPUT DENGAN MLFLOW.DATA.FROM_PANDAS()
            try:
                df_data = pd.read_csv(PREPROCESSED_DATA_PATH)
                
                # Menggunakan mlflow.data.from_pandas yang lebih ringkas dan sering berhasil
                # Menggunakan resolve() untuk memastikan absolute path dan skema file:///
                dataset = mlflow.data.from_pandas(
                    df=df_data,
                    source=f"file:///{PREPROCESSED_DATA_PATH.resolve()}", 
                    name="amazon_preprocessed_data"
                )
                mlflow.log_input(dataset, context="training")
                print(f"Dataset Preprocessed dicatat di MLflow di folder 'data/' dan sebagai 'Input' menggunakan mlflow.data.from_pandas.")
            except Exception as e:
                print(f"Gagal log dataset secara terstruktur. Error: {e}")
                
        else:
            print(f"Warning: Dataset Preprocessed tidak ditemukan di {PREPROCESSED_DATA_PATH.resolve()} untuk dicatat.")
        
        # a. Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Negative (0)', 'Positive (1)'], 
                    yticklabels=['Negative (0)', 'Positive (1)'])
        plt.title('Test Set Confusion Matrix')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close() 
        
        # b. Model Logging 
        mlflow.sklearn.log_model(
            sk_model=best_estimator,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME
        )
        
        # c. Log requirements.txt 
        req_path = "requirements.txt"
        if not os.path.exists(req_path):
             # Buat file jika tidak ada
             with open(req_path, 'w') as f:
                f.write("pandas\nnumpy\nscikit-learn\nmlflow\nmatplotlib\nseaborn")
        
        if os.path.exists(req_path):
            mlflow.log_artifact(req_path)
        
        run_id = run.info.run_id
        print(f"\nMLflow Run completed with ID: {run_id}")
        print(f"Model ({REGISTERED_MODEL_NAME}) dan artefak telah dicatat secara manual.")

def main():
    # --- SETUP MLFLOW ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000/") 
    # PERBAIKAN: Mengganti nama experiment untuk menghindari konflik "deleted state"
    EXPERIMENT_NAME = "Proyek Akhir Sentiment Analysis Tuning Ver2" 
    
    print("------------------------------------------------")
    print("Pastikan MLflow UI sudah berjalan di Terminal!")
    print("------------------------------------------------")
    
    try:
        # Coba set experiment. Jika gagal, buat yang baru.
        mlflow.set_experiment(EXPERIMENT_NAME)
    except MlflowException as e:
        # Perbaikan ini mengatasi RESOURCE_ALREADY_EXISTS dan Cannot set a deleted experiment
        print(f"MLflow Exception: Mencoba membuat/mengaktifkan experiment '{EXPERIMENT_NAME}'.")
        try:
            mlflow.create_experiment(EXPERIMENT_NAME)
            mlflow.set_experiment(EXPERIMENT_NAME)
        except Exception as e_new:
            print(f"Gagal membuat experiment baru: {e_new}")
            sys.exit(1)
    except Exception as e:
        print(f"Error saat mengatur MLflow Experiment: {e}")
        sys.exit(1)
        
    try:
        # 1. Load Data
        X_train, X_test, y_train, y_test = load_data(PREPROCESSED_DATA_PATH)
        
        # 2. Build and Tune Model (Randomized Search)
        print("\n--- Starting Hyperparameter Tuning (Randomized Search) ---")
        search = create_model_pipeline()
        # RandomizedSearchCV memiliki banyak output yang bisa menyebabkan error encoding, 
        # tetapi proses ini harus dijalankan
        search.fit(X_train, y_train) 
        
        print("\nBest Parameters Found:")
        for key, value in search.best_params_.items():
            print(f"- {key}: {value}")
        
        # 3. Evaluate and Log Model (Manual Logging)
        evaluate_and_log(search, X_test, y_test)
        
    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Pastikan Anda sudah menjalankan 'automate_Fathur-Rahman.py' terlebih dahulu.")
        sys.exit(1)
    except Exception as e:
        # Menangkap error lain yang mungkin terjadi
        if "'charmap' codec can't encode" not in str(e):
            print(f"\nAn unexpected error occurred: {e}")
            sys.exit(1)
        # Jika itu error charmap, kita bisa mengabaikannya di sini karena model sudah dilog
        else:
            print("\nWARNING: Error pencetakan konsol non-fatal diabaikan setelah MLflow logging.")

if __name__ == "__main__":
    main()
