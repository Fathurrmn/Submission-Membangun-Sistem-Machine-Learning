import pandas as pd
import os
import sys
from pathlib import Path

# Import tambahan untuk fallback split data
from sklearn.model_selection import train_test_split 

# Import pustaka Scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score 

# Import MLflow
import mlflow
import mlflow.sklearn
from mlflow.exceptions import MlflowException, RestException 
# Import untuk logging dataset terstruktur (best practice)
from mlflow.data.pandas_dataset import PandasDataset


# Path data yang sudah dipreproses
PREPROCESSED_DATA_PATH = Path('amazon_preprocessing') / 'amazon_preprocessed.csv'

# Set nama model yang akan didaftarkan di MLflow
REGISTERED_MODEL_NAME_BASIC = "Sentiment_LR_Basic"


# --- PERBAIKAN: Set encoding output konsol ke UTF-8 ---
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')
# ------------------------------------------------------


def load_data(path: Path):
    """Memuat data training dan testing dari file preprocessed."""
    full_path = path.resolve()
    print(f"Mencoba memuat data dari: {full_path}") 
    if not path.exists():
        raise FileNotFoundError(f"Data preprocessed tidak ditemukan di: {full_path}")

    df = pd.read_csv(path)
    
    if 'split' in df.columns:
        print("Kolom 'split' ditemukan. Memuat data berdasarkan kolom tersebut.")
        df_train = df[df['split'] == 'train']
        df_test = df[df['split'] == 'test']
        df_full = df 
    else:
        print("\n\n!! PERINGATAN FATAL: Kolom 'split' TIDAK DITEMUKAN di data preprocessed. !!")
        print("!! Data akan dibagi secara manual (fallback) menggunakan train_test_split (80/20).\n")
        
        X = df['review'].astype(str)
        y = df['sentiment']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        df_train = pd.DataFrame({'review': X_train, 'sentiment': y_train})
        df_test = pd.DataFrame({'review': X_test, 'sentiment': y_test})
        
        df_full = pd.concat([
            df_train.assign(split='train'),
            df_test.assign(split='test')
        ], ignore_index=True)


    X_train, y_train = df_train['review'].astype(str), df_train['sentiment']
    X_test, y_test = df_test['review'].astype(str), df_test['sentiment']
    
    print(f"Data Loaded: Train Samples={len(X_train)}, Test Samples={len(X_test)}")
    return X_train, X_test, y_train, y_test, df_full 

def create_and_train_model(X_train, y_train):
    """Mendefinisikan dan melatih model tanpa tuning."""
    
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000)), 
        ('clf', LogisticRegression(random_state=42, solver='liblinear', C=1.0)) 
    ])
    
    print("\n--- Training Model (Basic Logistic Regression) ---")
    pipeline.fit(X_train, y_train)
    return pipeline

def evaluate_model(model, X_test, y_test):
    """Mengevaluasi model pada test set."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] 
    
    metrics = {
        "test_f1_score": f1_score(y_test, y_pred, zero_division=0),
        "test_roc_auc": roc_auc_score(y_test, y_proba)
    }
    
    print("--- Model Evaluation (Test Set) ---")
    print(f"F1-Score: {metrics['test_f1_score']:.4f}")
    print(f"ROC-AUC: {metrics['test_roc_auc']:.4f}")
    
    return metrics

def main():
    # --- SETUP MLFLOW: MENGGUNAKAN HTTP TRACKING UNTUK KOMUNIKASI SERVER ---
    TRACKING_URI = "http://127.0.0.1:5000/" 
    
    try:
        # 1. Set Tracking URI ke server HTTP
        mlflow.set_tracking_uri(TRACKING_URI)
        # 2. Set Registry URI ke server HTTP
        mlflow.set_registry_uri(TRACKING_URI) 
        print(f"MLflow Tracking & Registry URI diatur ke: {TRACKING_URI}")
        
    except Exception as e:
        print(f"Fatal: Gagal mengatur MLflow URI: {e}")
        sys.exit(1)


    # Mengatasi error 'deleted state'
    experiment_name = "Proyek Akhir Sentiment Analysiss"
    try:
        mlflow.set_experiment(experiment_name)
    except MlflowException:
        try:
            mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception:
            print("Fatal: Gagal membuat/mengaktifkan experiment. Pastikan tidak ada konflik nama.")
            sys.exit(1)


    # =================================================================
    # AKTIVASI AUTOLOGGING MLFLOW
    # =================================================================
    mlflow.sklearn.autolog(log_models=True)
    
    print("------------------------------------------------")
    print("MLflow Autologging AKTIF. Parameter dan Metrik Dasar akan dicatat otomatis.")
    print("------------------------------------------------")
    
    try:
        # 1. Load Data
        X_train, X_test, y_train, y_test, df_full = load_data(PREPROCESSED_DATA_PATH) 
        
        # 2. Siapkan Dataset Terstruktur untuk logging
        print("Mempersiapkan dataset terstruktur (Kolom 'Dataset')...")
        # Menggunakan Path.resolve() untuk mendapatkan path absolut yang stabil
        dataset = mlflow.data.from_pandas(
            df=df_full,
            source=f"file:///{PREPROCESSED_DATA_PATH.resolve().as_posix()}", # Gunakan .as_posix() untuk path yang kompatibel di MLflow
            name="amazon_preprocessed_basic"
        )
        
        # 3. Start MLflow Run dan Latih Model
        with mlflow.start_run() as run:
            # --- LOG INPUT DATASET SECARA MANUAL (BEST PRACTICE) ---
            mlflow.log_input(dataset, context="training")
            print("Dataset dicatat sebagai input terstruktur.")

            # --- TRAINING (Autologging bekerja di sini) ---
            model = create_and_train_model(X_train, y_train)
            
            # --- EVALUASI ---
            metrics = evaluate_model(model, X_test, y_test)
            mlflow.log_metrics(metrics)

            # --- LOG ARTEFAK MANUAL (untuk file pendukung) ---
            if os.path.exists(PREPROCESSED_DATA_PATH):
                mlflow.log_artifact(str(PREPROCESSED_DATA_PATH), artifact_path="data")
                print(f"File Dataset Preprocessed dicatat sebagai artefak di folder 'data/'.")
            
            print(f"\nMLflow Run completed with ID: {run.info.run_id}")
            
            # Catat model ke Model Registry (Wajib dilakukan manual)
            mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/model",
                name=REGISTERED_MODEL_NAME_BASIC
            )
            print(f"Model didaftarkan ke Model Registry sebagai '{REGISTERED_MODEL_NAME_BASIC}'")


    except FileNotFoundError as e:
        print(f"\nFATAL ERROR: {e}")
        print("Pastikan Anda sudah menjalankan preprocessing dan file data ada di lokasi yang benar.")
        sys.exit(1)
    except RestException as e:
        # Menangkap error koneksi MLflow (e.g., failed to connect to http://127.0.0.1:5000)
        print("\nFATAL ERROR: Koneksi MLflow Gagal atau Server Tidak Terkonfigurasi dengan Benar.")
        print(f"Detail: {e}")
        print("Pastikan Anda sudah menjalankan server MLflow UI di terminal SEBELUM menjalankan skrip ini.")
        
        # --- SOLUSI PENTING: Perintah yang harus dijalankan pengguna ---
        # Menggunakan path absolut Windows dengan quote untuk menangani spasi
        mlruns_path = 'C:/Users/FathurNitro/OneDrive/Documents/sub dicoding/membangun machine learning bismillah/membangun model/mlruns'
        
        print("\n---------------------------------------------------------------------------------------------------------------------------------")
        print("!! TINDAKAN WAJIB !!")
        print(f"Untuk memastikan hasil tercatat di '{mlruns_path}', Anda HARUS menjalankan server MLflow UI dengan perintah berikut:")
        print(f"\nmlflow ui --backend-store-uri \"file:{mlruns_path}\"")
        print("\nSetelah server berjalan, ulangi eksekusi skrip Python ini.")
        print("---------------------------------------------------------------------------------------------------------------------------------")
        sys.exit(1)
    except Exception as e:
        if "'charmap' codec can't encode" not in str(e):
             print(f"\nAn unexpected error occurred: {e}")
        else:
            print(f"\nWARNING: Non-fatal encoding error caught. Script finished, check MLflow for results.")
            
if __name__ == "__main__":
    main()