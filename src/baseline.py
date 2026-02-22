import argparse
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error
import warnings
import boto3
import botocore
import io

# Ignora i fastidiosi warning di fsspec/s3fs
warnings.filterwarnings("ignore", category=UserWarning, module="fsspec")

def save_baseline_metrics(dataset, trees, max_samples, train_time, inf_time, metrics_dict, target_bucket):
    s3_client = boto3.client('s3')
    
    # Percorso dinamico
    s3_key = f"results/{dataset}/baseline_{dataset}_result.csv"
    
    new_row_df = pd.DataFrame([{
        'Dataset': dataset, 
        'Trees': trees,
        'Max_Samples': max_samples, # Utile per ricordare quante righe la singola macchina è riuscita a reggere
        'Train_Time': round(train_time, 2), 
        'Infer_Time': round(inf_time, 2), 
        'Metrics': str(metrics_dict)
    }])

    try:
        # Tenta di scaricare il CSV esistente da S3
        obj = s3_client.get_object(Bucket=target_bucket, Key=s3_key)
        df_existing = pd.read_csv(io.BytesIO(obj['Body'].read()))
        # Accoda (APPEND)
        df_final = pd.concat([df_existing, new_row_df], ignore_index=True)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchKey':
            # Se non esiste, crea il dataframe da zero
            df_final = new_row_df
        else:
            print(f"!! Errore imprevisto di S3 durante il salvataggio: {e}")
            return
            
    # Salva il file aggiornato su S3
    csv_buffer = io.StringIO()
    df_final.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=target_bucket, Key=s3_key, Body=csv_buffer.getvalue())
    print(f">> Risultati baseline accodati in: s3://{target_bucket}/{s3_key}")


def run_baseline(args):
    print(f"\n{'='*50}")
    print(f" AVVIO BASELINE SCALARE SU SINGOLA MACCHINA ({args.dataset.upper()})")
    print(f"{'='*50}")

    # 1. Caricamento Dati di Training
    print(f"-> Caricamento Train Set da: {args.train_path} (Max {args.max_samples} righe)...")
    df_train = pd.read_csv(args.train_path, dtype=np.float32, nrows=args.max_samples)
    
    target_col = 'Label' if 'Label' in df_train.columns else df_train.columns[-1]

    if args.dataset == 'higgs':
        df_train[target_col] = df_train[target_col].astype(np.int8)

    X_train = df_train.drop(target_col, axis=1).values
    y_train = df_train[target_col].values

    # 2. Caricamento Dati di Test
    print(f"-> Caricamento Test Set da: {args.test_path}...")
    df_test = pd.read_csv(args.test_path, dtype=np.float32)
    
    if args.dataset == 'higgs':
        df_test[target_col] = df_test[target_col].astype(np.int8)

    X_test = df_test.drop(target_col, axis=1).values
    y_test = df_test[target_col].values

    # 3. Modello Scikit-Learn
    if args.dataset == 'taxi':
        model = RandomForestRegressor(n_estimators=args.trees, n_jobs=-1, random_state=42)
    else:
        model = RandomForestClassifier(n_estimators=args.trees, n_jobs=-1, random_state=42)

    # 4. Addestramento
    print(f"\n-> Inizio Addestramento di {args.trees} alberi su {len(X_train)} campioni...")
    start_train = time.time()
    model.fit(X_train, y_train)
    train_duration = time.time() - start_train
    print(f"   [OK] Training completato in: {train_duration:.2f}s")

    # 5. Inferenza
    print(f"\n-> Inizio Inferenza su {len(X_test)} campioni...")
    start_inf = time.time()
    preds = model.predict(X_test)
    inf_duration = time.time() - start_inf
    print(f"   [OK] Inferenza completata in: {inf_duration:.2f}s")
    print(f"   [OK] Throughput: {len(preds) / inf_duration:.0f} pred/sec")

    # 6. Metriche
    metrics_dict = {}
    print(f"\n{'-'*30}\n METRICHE FINALI ({args.dataset.upper()})\n{'-'*30}")
    if args.dataset == 'taxi':
        mse = mean_squared_error(y_test, preds)
        rmse = float(np.sqrt(mse)) # Casting a float per evitare problemi di serializzazione
        mae = mean_absolute_error(y_test, preds)
        metrics_dict = {'MSE': round(mse, 4), 'RMSE': round(rmse, 4), 'MAE': round(mae, 4)}
        print(f"MSE:  {mse:.4f}\nRMSE: {rmse:.4f}\nMAE:  {mae:.4f}")
    else:
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='weighted')
        metrics_dict = {'Accuracy': round(acc, 4), 'F1-Score': round(f1, 4)}
        print(f"Accuracy: {acc*100:.2f}%\nF1-Score: {f1:.4f}")
    print("="*50 + "\n")

    # 7. Salvataggio S3
    save_baseline_metrics(
        args.dataset, args.trees, args.max_samples, 
        train_duration, inf_duration, metrics_dict, args.s3_bucket
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Random Forest Single Machine")
    parser.add_argument('--dataset', type=str, required=True, choices=['taxi', 'higgs'])
    parser.add_argument('--train_path', type=str, required=True, help="Percorso S3 o locale del file di training")
    parser.add_argument('--test_path', type=str, required=True, help="Percorso S3 o locale del file di test")
    parser.add_argument('--trees', type=int, default=50, help="Numero di alberi della foresta")
    parser.add_argument('--max_samples', type=int, default=500000, help="Limite righe lette per evitare OOM")
    # Nuovo parametro per specificare il bucket dinamicamente
    parser.add_argument('--s3_bucket', type=str, default='distributed-random-forest-bkt', help="Nome del bucket S3")

    args = parser.parse_args()
    run_baseline(args)
