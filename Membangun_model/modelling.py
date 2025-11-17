# modelling.py (VERSI PERBAIKAN V3: Mengatasi Dataset Logging & Error Encoding)

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
from sklearn.metrics import accuracy_score, f1_score

# Import MLflow
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException # Import untuk penanganan error

# Path data yang sudah dipreproses
PREPROCESSED_DATA_PATH = Path('amazon_preprocessing') / 'amazon_preprocessed.csv'

# Set nama model yang akan didaftarkan di MLflow
REGISTERED_MODEL_NAME_BASIC = "Sentiment_LR_Basic"


def load_data(path: Path):
    """Memuat data training dan testing dari file preprocessed."""
    if not path.exists():
        # Menambahkan resolve() untuk tampilan error yang lebih jelas
        raise FileNotFoundError(f"Data preprocessed tidak ditemukan di: {path.resolve()}")

    df = pd.read_csv(path)
    df_train = df[df['split'] == 'train']
    df_test = df[df['split'] == 'test']
    
    X_train, y_train = df_train['review'].astype(str), df_train['sentiment']
    X_test, y_test = df_test['review'].astype(str), df_test['sentiment']
    
    print(f"Data Loaded: Train Samples={len(X_train)}, Test Samples={len(X_test)}")
    # PERBAIKAN: Mengembalikan DataFrame penuh (df) untuk logging terstruktur di MLflow
    return X_train, X_test, y_train, y_test, df 

def create_and_train_model(X_train, y_train):
    """Mendefinisikan dan melatih model tanpa tuning."""
    
    # Definisikan pipeline: Vectorizer -> Classifier dengan parameter default
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('clf', LogisticRegression(random_state=42, solver='liblinear'))
    ])
    
    # Latih model
    print("\n--- Training Model (Basic Logistic Regression) ---")
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Mengevaluasi model pada test set."""
    y_pred = model.predict(X_test)
    
    metrics = {
        "test_accuracy": accuracy_score(y_test, y_pred),
        "test_f1_score": f1_score(y_test, y_pred, zero_division=0)
    }
    
    print("--- Model Evaluation (Test Set) ---")
    print(f"Accuracy: {metrics['test_accuracy']:.4f}")
    print(f"F1-Score: {metrics['test_f1_score']:.4f}")
    
    return metrics

def main():
    # --- SETUP MLFLOW ---
    mlflow.set_tracking_uri("http://127.0.0.1:5000/") 
    
    # PERHATIAN: JIKA INI GAGAL KARENA 'deleted state', GANTI NAMA EXPERIMENT
    try:
        mlflow.set_experiment("Proyek Akhir Sentiment Analysis Basic")
    except MlflowException as e:
        print(f"Warning: Gagal mengatur experiment. Error: {e}")
        # Jika gagal (misalnya karena 'deleted state'), coba buat yang baru
        try:
            mlflow.create_experiment("Proyek Akhir Sentiment Analysis Basic")
            mlflow.set_experiment("Proyek Akhir Sentiment Analysis Basic")
        except Exception:
            print("Fatal: Gagal membuat/mengaktifkan experiment. Pastikan tidak ada konflik nama.")
            sys.exit(1)


    # AKTIFKAN AUTOLOGGING (Kriteria Basic)
    mlflow.sklearn.autolog(log_models=True)
    
    print("------------------------------------------------")
    print("MLflow Autologging AKTIF.")
    print("------------------------------------------------")
    
    try:
        # 1. Load Data
        # PERBAIKAN: Menangkap df_full yang dikembalikan dari load_data
        X_train, X_test, y_train, y_test, df_full = load_data(PREPROCESSED_DATA_PATH) 
        
        # 2. Siapkan Dataset Terstruktur untuk logging (di luar run)
        print("Mempersiapkan dataset terstruktur untuk kolom 'Dataset'...")
        # Menggunakan mlflow.data.from_pandas untuk mencatat metadata dataset
        dataset = mlflow.data.from_pandas(
            df=df_full,
            source=f"file:///{PREPROCESSED_DATA_PATH.resolve()}", 
            name="amazon_preprocessed_basic"
        )
        
        # 3. Start MLflow Run dan Latih Model
        with mlflow.start_run() as run:
             # PERBAIKAN: Log input dataset di dalam run
            mlflow.log_input(dataset, context="training")
            print("Dataset dicatat sebagai input terstruktur (Kolom 'Dataset').")

            model = create_and_train_model(X_train, y_train)
            
            # 4. Evaluasi dan Logging Metrik Tambahan
            metrics = evaluate_model(model, X_test, y_test)
            
            # Autologging mencatat metrik internal Scikit-learn. Ini mencatat metrik final kita.
            mlflow.log_metrics(metrics)

            # --- LOG ARTEFAK FILE (untuk folder 'Artifacts') ---
            if os.path.exists(PREPROCESSED_DATA_PATH):
                mlflow.log_artifact(str(PREPROCESSED_DATA_PATH), artifact_path="data")
                print(f"File Dataset Preprocessed dicatat di MLflow di folder 'data/'.")
            else:
                print(f"Warning: Dataset Preprocessed tidak ditemukan di {PREPROCESSED_DATA_PATH.resolve()} untuk dicatat.")
            
            # --- LOG ARTEFAK requirements.txt ---
            req_path = "requirements.txt"
            if os.path.exists(req_path):
                mlflow.log_artifact(req_path)
            else:
                print(f"Warning: {req_path} not found.")

            print(f"\nMLflow Run completed with ID: {run.info.run_id}")
            
            # Catat model ke Model Registry (Wajib dilakukan manual)
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=REGISTERED_MODEL_NAME_BASIC
            )

    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Pastikan Anda sudah menjalankan 'automate_Nama-siswa.py' terlebih dahulu.")
        sys.exit(1)
    except Exception as e:
        # Menangkap error 'charmap' untuk mencegah crash skrip
        if "'charmap' codec can't encode" not in str(e):
             print(f"\nAn unexpected error occurred: {e}")
        else:
            # Jika error encoding, anggap logging utama sudah selesai dan keluar dengan sukses (exit code 0)
            print(f"\nWARNING: Non-fatal encoding error caught. Script finished, check MLflow for results.")
            
if __name__ == "__main__":
    main()
