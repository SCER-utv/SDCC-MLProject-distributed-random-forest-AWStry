import os
import sys
import argparse  # [MODIFICA 1] Import per parametri da terminale
import csv       # [MODIFICA 2] Import per salvare le metriche
from concurrent.futures import ThreadPoolExecutor
import time
import gc
import json
import pandas as pd
import numpy as np # [MODIFICA] Serve per gestire il float32
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.factories.higgs_task_factory import HiggsTaskFactory
from src.network.grpc_master import GrpcMaster
from src.core.factories.ids_task_factory import IDSTaskFactory
from src.core.factories.taxi_task_factory import TaxiTaskFactory
import io
from src.utils.config import load_config
import botocore 
import boto3
import warnings

def save_metrics(dataset, n_workers, n_trees, strategy_name, train_time, inf_time, metrics_dict, config):
    s3_client = boto3.client('s3')
    target_bucket = config.get("s3_bucket", "distributed-random-forest-bkt")
    
    # [MODIFICA S3] Creazione percorso dinamico: es. "results/higgs/higgs_results.csv"
    s3_key = f"results/{dataset}/{dataset}_results.csv"
    
    new_row_df = pd.DataFrame([{
        'Dataset': dataset, 
        'Workers': n_workers, 
        'Trees': n_trees, 
        'Strategy': strategy_name, 
        'Train_Time': round(train_time, 2), 
        'Infer_Time': round(inf_time, 2), 
        'Metrics': str(metrics_dict)
    }])

    try:
        # Tenta di scaricare il CSV esistente da S3
        obj = s3_client.get_object(Bucket=target_bucket, Key=s3_key)
        df_existing = pd.read_csv(io.BytesIO(obj['Body'].read()))
        # Se esiste, accoda la nuova riga (APPEND continuo)
        df_final = pd.concat([df_existing, new_row_df], ignore_index=True)
        
    except botocore.exceptions.ClientError as e:
        # Se il file non esiste (errore 404 NoSuchKey), crea il dataframe partendo dalla nuova riga
        if e.response['Error']['Code'] == 'NoSuchKey':
            df_final = new_row_df
        else:
            print(f"!! Errore imprevisto di S3 durante il salvataggio: {e}")
            return
            
    # Salva il file aggiornato su S3 (sovrascrivendo la vecchia versione con quella accodata)
    csv_buffer = io.StringIO()
    df_final.to_csv(csv_buffer, index=False)
    s3_client.put_object(Bucket=target_bucket, Key=s3_key, Body=csv_buffer.getvalue())
    print(f">> Risultati accodati permanentemente in: s3://{target_bucket}/{s3_key}")

if __name__ == '__main__':

    # [MODIFICA 4] Configurazione di Argparse
    parser = argparse.ArgumentParser(description="Distributed Random Forest Master")
    parser.add_argument('--dataset', type=str, required=True, choices=['taxi', 'higgs', 'ids', 'covertype'])
    parser.add_argument('--workers', nargs='+', required=True, help="Lista IP:Porta dei worker")
    parser.add_argument('--trees', type=int, default=10, help="Numero totale di alberi")
    # [MODIFICA] Aggiungiamo il percorso del file strategie
    parser.add_argument('--strategy_file', type=str, default='config/worker_strategies.json')
    
    args = parser.parse_args()
    num_active_workers = len(args.workers)

    # [MODIFICA] BATCH SIZE DINAMICO (HARDCODED, DA MODIFICARE!)
    if args.dataset == 'taxi':
        BATCH_SIZE = 50000  # Limite di sicurezza per la regressione
    elif args.dataset == 'higgs':
        BATCH_SIZE = 35000  # Limite di sicurezza per la classificazione
    else:
        BATCH_SIZE = 20000  # Default di fallback

    config = load_config()

    try:
        # [MODIFICA LETTURA JSON BIFORCATA]
        task_category = "regression" if args.dataset == 'taxi' else "classification"
        with open(args.strategy_file, 'r') as f:
            full_strategy_map = json.load(f)

        str_num = str(num_active_workers)
        if str_num in full_strategy_map[task_category]:
            config['worker_strategies'] = full_strategy_map[task_category][str_num]
            print(f">> Caricato set di {num_active_workers} strategie ({task_category}).")
        else:
            print(f"!! ERRORE: Nessuna config specifica per {num_active_workers} worker in {task_category}.")
            sys.exit(1)
            
    except Exception as e:
        print(f"!! ERRORE NELLA LETTURA DEL JSON: {e}")
        sys.exit(1)

    # [MODIFICA 5] INIEZIONE DEI PARAMETRI DINAMICI NELLA CONFIG IN RAM
    config['num_workers'] = len(args.workers)
    config['workers'] = args.workers
    config['total_trees'] = args.trees

    # Selezione Factory
    if args.dataset == 'taxi': factory = TaxiTaskFactory()
    elif args.dataset == 'higgs': factory = HiggsTaskFactory()
    else: factory = IDSTaskFactory()

    strategy = factory.create_strategy()
    data_manager = factory.create_data_manager(strategy)

    # Aggiorniamo il bit di task nel config per i messaggi gRPC
    config['task'] = strategy.get_task_type()

    # 1. Preparazione Dati (Ora restituisce solo il percorso!)
    test_path = data_manager.prepare_data(config)
    config['dataset_path'] = data_manager.get_shards_path(config)
    print(f">> SHARDING COMPLETATO: I worker useranno -> {config['dataset_path']}")

    # 2. Setup Master (Iniettiamo la strategia nel costruttore)
    grpc_master = GrpcMaster(config, strategy)
    grpc_master.connect()

    # 3. Training Distribuito
    # Ora master.train() distribuirà subforest_id e base_seed a ogni worker
    print("\n--- Avvio Training Distribuito ---")
    start_train = time.time()
    grpc_master.train()
    train_duration = time.time() - start_train
    print(f"Training completato in: {train_duration:.2f}s")

    # 4. Inferenza (Testing)
    # Otteniamo il nome del target dinamicamente dal data_manager specifico
    target_col = data_manager.get_target_column()

    # [MODIFICA AWS] Costruiamo l'URI S3 del Test Set in modo dinamico in base al dataset scelto
    target_bucket = config.get("s3_bucket", "distributed-random-forest-bkt")
    
    # Il nome del file dipende dal dataset (es: "taxi_test_set.csv", "higgs_test_set.csv")
    s3_test_path = f"s3://{target_bucket}/temp/{args.dataset}_test_set.csv"
    
    print(f"Caricamento Test Set da S3: {s3_test_path}...")
    
    # Aggiungiamo nrows=50000 per evitare che il Master vada in Out-Of-Memory!
    test_df = pd.read_csv(s3_test_path, dtype=np.float32)
    
    if config['task'] != 1: # Se è classificazione, la label è int
        test_df[target_col] = test_df[target_col].astype(np.int8)

    X_test = test_df.drop(target_col, axis=1).values.astype(np.float32) 
    y_true = test_df[target_col].values
    total_samples = len(X_test)

    # [MODIFICA 3: CHUNKING EFFICIENTE]
    # Creiamo i chunk affettando numpy (veloce) e convertendo in lista solo il pezzettino che serve
    # Questo mantiene l'uso della RAM basso.
    chunks = []
    for i in range(0, total_samples, BATCH_SIZE):
        chunk_np = X_test[i : i + BATCH_SIZE]
        chunks.append(chunk_np.tolist()) # Convertiamo solo il pacchetto da inviare

    print(f"\n--- Inferenza su {total_samples} campioni ---")
    print(f"Batch Size: {BATCH_SIZE} | Chunk Totali: {len(chunks)}")

    start_predict = time.time()
    results = []
    processed_count = 0

    # Parallelizzazione invio chunk
    with ThreadPoolExecutor(max_workers=10) as ex:
        # predict_batch userà strategy.aggregate per media o moda
        for batch_res in ex.map(grpc_master.predict_batch, chunks):
            results.extend(batch_res)

            processed_count += len(batch_res)
            if processed_count % 100000 == 0:
                perc = (processed_count / total_samples) * 100
                print(f"[{time.strftime('%H:%M:%S')}] Progresso: {processed_count}/{total_samples} ({perc:.1f}%)")

    duration = time.time() - start_predict
    grpc_master.close()

    valid_preds = [p for p in results if p is not None]

    # [MODIFICA] Allineamento y_true con i risultati validi
    # (Attenzione: se results ha dei None, y_true deve essere filtrato allo stesso modo)
    if len(valid_preds) < len(y_true):
         # Ricostruiamo valid_true solo se ci sono stati fallimenti
         valid_true = []
         for i, res in enumerate(results):
             if res is not None:
                 valid_true.append(y_true[i])
    else:
         valid_true = y_true 
         
    print(f"\n--- REPORT FINALE ---")
    print(f"Campioni Totali: {total_samples}")
    print(f"Predizioni Ricevute: {len(valid_preds)}")
    print(f"Tempo Inferenza: {duration:.4f}s")

    if duration > 0:
        print(f"Throughput: {len(valid_preds) / duration:.0f} pred/sec")

    if len(valid_preds) > 0:
        # [MODIFICA IMPORTANTE QUI]
        # 1. Assegniamo il risultato di report() a una variabile 'metrics'
        metrics = strategy.report(valid_true, valid_preds)
        
        # 2. Salviamo tutto nel CSV per i grafici!
        # (Presuppone che train_duration sia stato calcolato prima)
        save_metrics(args.dataset, config['num_workers'], args.trees, "JSON_Strategy", train_duration, duration, metrics, config)
        print(">> Risultati salvati in experiment_results.csv per l'analisi!")
    else:
        print("Errore: Nessuna predizione ricevuta dai Worker.")
