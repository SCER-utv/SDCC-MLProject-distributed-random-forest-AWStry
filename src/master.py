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

from src.utils.config import load_config

# [MODIFICA 1] Batch Size aumentato per ridurre overhead di rete (OK)
BATCH_SIZE = 20000

# [MODIFICA 3] Funzione per salvare i risultati per i grafici dell'esame
def save_metrics(dataset, n_workers, n_trees, max_depth, train_time, inf_time, metrics_dict):
    file_exists = os.path.isfile('experiment_results.csv')
    with open('experiment_results.csv', mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Dataset', 'Workers', 'Trees', 'Max_Depth', 'Train_Time', 'Infer_Time', 'Metrics'])
        writer.writerow([dataset, n_workers, n_trees, max_depth, train_time, inf_time, str(metrics_dict)])

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

    with open(args.strategy_file, 'r') as f:
        full_strategy_map = json.load(f)

    config = load_config()

    try:
        with open(args.strategy_file, 'r') as f:
            full_strategy_map = json.load(f)

        str_num = str(num_active_workers)
        if str_num in full_strategy_map:
            # Salviamo nella config SOLO la lista per il numero esatto di worker
            config['worker_strategies'] = full_strategy_map[str_num]
            print(f">> Caricato set di {num_active_workers} strategie ottimizzate dal JSON.")
        else:
            print(f"!! ERRORE: Nessuna config specifica per {num_active_workers} worker nel file JSON.")
            sys.exit(1) # Crash pilotato
            
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

    # [MODIFICA 6] Caricamento ottimizzato del solo file di Test
    print(f"Caricamento Test Set da: {test_path}...")
    test_df = pd.read_csv(test_path, dtype=np.float32)
    
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
        save_metrics(args.dataset, config['num_workers'], args.trees, "JSON_Strategy", train_duration, duration, metrics)
        print(">> Risultati salvati in experiment_results.csv per l'analisi!")
    else:
        print("Errore: Nessuna predizione ricevuta dai Worker.")
