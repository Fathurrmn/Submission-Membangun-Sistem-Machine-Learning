import requests
import json
import time

# --- KONFIGURASI PENGUJIAN ---
# URL endpoint FastAPI Anda (pastikan main.py sedang berjalan di port 8000)
API_URL = "http://localhost:8000/predict"

# Contoh data transaksi baru (Input harus sesuai dengan class Transaction di main.py)
# Data ini akan dikirim dalam format JSON ke endpoint /predict.
NEW_TRANSACTION_DATA = {
    "CustomerID": 999999,
    "ProductID": "A",
    "Quantity": 5,
    "Price": 85.00,
    "TransactionDate": "2025-12-01 10:30:00",
    "PaymentMethod": "Credit Card",
    "StoreLocation": "2020 Maple Ave, Chicago, IL 60601",
    "ProductCategory": "Electronics",
    # Perhatikan: Nama field disesuaikan dengan skema Pydantic
    "DiscountApplied_percent": 15.5 
}

def test_prediction_endpoint(data):
    """Mengirim permintaan POST ke endpoint /predict dan mencetak hasilnya."""
    
    headers = {'Content-Type': 'application/json'}
    print(f"Mengirim permintaan ke: {API_URL}")
    print(f"Data yang dikirim: {json.dumps(data, indent=4)}\n")
    
    start_time = time.time()
    try:
        # Kirim permintaan POST
        response = requests.post(API_URL, headers=headers, data=json.dumps(data))
        end_time = time.time()
        
        # Cek status code
        if response.status_code == 200:
            result = response.json()
            print("="*40)
            print("✅ PREDIKSI BERHASIL (Status 200 OK)")
            print(f"Waktu Respon Jaringan: {round(end_time - start_time, 4)} detik")
            print("--- Hasil Prediksi ---")
            print(json.dumps(result, indent=4))
            print("="*40)
            
            # Catatan: Latensi model juga dicatat di body respons
            if 'latency_seconds' in result:
                print(f"Latensi Model (dihitung FastAPI): {result['latency_seconds']} detik")
                
        elif response.status_code == 503:
             print("="*40)
             print(f"❌ ERROR SERVER: Status 503 Service Unavailable")
             print(f"Penyebab: Model belum dimuat. Pastikan '{MLFLOW_MODEL_URI}' valid dan bisa diakses.")
             print("="*40)

        else:
            print("="*40)
            print(f"❌ ERROR HTTP: Status {response.status_code}")
            print("Respons Server:")
            print(response.text)
            print("="*40)

    except requests.exceptions.ConnectionError:
        print("="*60)
        print("❌ ERROR KONEKSI: Gagal terhubung ke server FastAPI.")
        print("Pastikan Anda menjalankan skrip 'main.py' di terminal lain:")
        print("uvicorn main:app --host 0.0.0.0 --port 8000")
        print("="*60)
    except Exception as e:
        print(f"Terjadi kesalahan tak terduga: {e}")

if __name__ == "__main__":
    test_prediction_endpoint(NEW_TRANSACTION_DATA)