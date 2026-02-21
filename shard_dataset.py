import pandas as pd
import numpy as np
import os
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split
import argparse
import json
import sys

# Aggiungiamo la root del progetto al path per poter importare 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

### [INIZIO MODIFICA AWS] Utilizzo della utility ufficiale config.py ###
from src.utils.config import load_config

# Carichiamo la configurazione usando il tuo modulo centralizzato
CONFIG = load_config()

# Estraiamo dinamicamente il bucket e i percorsi dal JSON
S3_BUCKET = CONFIG.get("s3_bucket") 
PATHS = CONFIG.get("paths")

if not S3_BUCKET or not PATHS:
    raise ValueError("Il file config.json deve contenere 's3_bucket' e 'paths'.")
### [FINE MODIFICA AWS] ###

def shard_dataset(name, config_paths, n_shards, strategy):

    ### [INIZIO MODIFICA AWS] Gestione dinamica dei percorsi S3 ###
    input_path = config_paths["full"]
    # Rimuoviamo l'output_dir locale. Usiamo il prefisso S3 dal config!
    s3_prefix = config_paths["shards_prefix"] 
    
    # Il test set va direttamente su S3
    test_set_path = f"s3://{S3_BUCKET}/temp/{name}_test_set.csv"
    ### [FINE MODIFICA AWS] ###
    
    print(f"\n[{name.upper()}] Inizio procedura di Sharding Intelligente...")
    print(f" -> Cerco il file: {input_path}")

    if not os.path.exists(input_path):
        print(f" -> ERRORE: File non trovato. Salto il dataset {name}.")
        return

    is_regression = strategy
    tipo_task = "REGRESSIONE (KFold)" if is_regression else "CLASSIFICAZIONE (StratifiedKFold)"
    print(f" -> Task: {tipo_task} | Numero di Shards: {n_shards}")
    
    # Variabili per tenere traccia dei totali
    total_rows = 0

    # Legge mezzo milione di righe alla volta (RAM sicura!)
    chunk_size = 500000 

    # Leggiamo il file a blocchi
    for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size, dtype=np.float32)):
        
        # 1. Mischiamo le righe all'interno del blocco
        chunk = chunk.sample(frac=1, random_state=42).reset_index(drop=True)
        total_rows += len(chunk)
        
        # [MODIFICA] ESTRAZIONE DEL TEST SET (20%) DAL CHUNK ---
        if is_regression:
            train_chunk, test_chunk = train_test_split(chunk, test_size=0.20, random_state=42)
        else:
            y_chunk_for_split = chunk['Label'].astype(int)
            train_chunk, test_chunk = train_test_split(chunk, test_size=0.20, random_state=42, stratify=y_chunk_for_split)

        # Salviamo/Accodiamo il Test Set nel suo file
        test_mode = 'w' if chunk_idx == 0 else 'a'
        test_header = True if chunk_idx == 0 else False
        test_chunk.to_csv(test_set_path, mode=test_mode, header=test_header, index=False)
        
        # SHARDING INTELLIGENTE SULL' 80% RIMANENTE (train_chunk) 
        # [MODIFICA] Usiamo train_chunk invece di chunk per creare i frammenti dei worker
        if is_regression:
            splitter = KFold(n_splits=n_shards, shuffle=True, random_state=42)
            gen = splitter.split(train_chunk)
        else:
            y_train_chunk = train_chunk['Label'].astype(int)
            splitter = StratifiedKFold(n_splits=n_shards, shuffle=True, random_state=42)
            gen = splitter.split(train_chunk, y_train_chunk)

        # Salviamo i risultati del generatore (gen) nei rispettivi file
        for i, (_, test_index) in enumerate(gen):
            
            # test_index contiene gli indici delle righe per questo shard
            split_df = train_chunk.iloc[test_index]
            
            ### [INIZIO MODIFICA AWS] Costruzione dinamica dell'URI per lo shard S3 ###
            # Esempio: s3://distributed-random-forest-bkt/shards/taxi/train_part_1.csv
            full_path = f"{s3_prefix}{i+1}.csv"
            ### [FINE MODIFICA AWS] ###
            
            mode = 'w' if chunk_idx == 0 else 'a'
            header = True if chunk_idx == 0 else False
            
            split_df.to_csv(full_path, mode=mode, header=header, index=False)

        print(f" Elaborate e stratificate {total_rows} righe totali...")

    print(f"\n[{name.upper()}] Sharding completato! Create {n_shards} partizioni perfette su S3.")

if __name__ == "__main__":

# Configurazione dei parametri da terminale
    parser = argparse.ArgumentParser(description="Utility per lo Sharding dei Dataset (Job Pre-Flight)")
    parser.add_argument('--dataset', type=str, required=True, choices=['taxi', 'higgs', 'ids', 'covertype'])
    parser.add_argument('--shards', type=int, required=True, help="Numero di partizioni (deve coincidere col numero di Worker!)")
    
    args = parser.parse_args()

    # Determiniamo se è regressione (solo taxi) o classificazione (gli altri)
    # Questa logica sostituisce il parametro booleano "strategy"
    is_regression = True if args.dataset == 'taxi' else False

    # Lanciamo lo sharding con i parametri dinamici!
    shard_dataset(args.dataset, PATHS[args.dataset], n_shards=args.shards, strategy=is_regression)