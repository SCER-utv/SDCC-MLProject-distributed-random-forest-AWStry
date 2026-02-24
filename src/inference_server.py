import os
import sys
import json
import boto3
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from urllib.parse import urlparse

# Aggiungiamo la root del progetto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importiamo le tue strategie per riutilizzare la logica di aggregazione
from src.core.strategies.classification_strategy import ClassificationStrategy
from src.core.strategies.regression_strategy import RegressionStrategy

app = Flask(__name__)

# Variabili globali per mantenere lo stato in RAM
loaded_subforests = []
current_strategy = None
model_feature_count = 0

# Scarica tutte le parti del modello da S3 e le carica in RAM
def download_and_load_model(model_id, dataset_type):
    
    global loaded_subforests, current_strategy, model_feature_count
    
    # Pulizia precedente se stiamo caricando un nuovo modello
    loaded_subforests.clear()
    
    # 1. Setup AWS e Strategia
    s3_client = boto3.client('s3')
    bucket_name = os.getenv('AWS_S3_BUCKET', 'distributed-random-forest-bkt')
    
    # La cartella su S3 dipende dal dataset (taxi o higgs/ids)
    s3_folder = "taxi" if dataset_type == "taxi" else "higgs"
    prefix = f"models/{s3_folder}/{model_id}"
    
    # Decidiamo la strategia in base al dataset
    if dataset_type == "taxi":
        current_strategy = RegressionStrategy()
        print(">> Modalità: REGRESSIONE")
    else:
        current_strategy = ClassificationStrategy()
        print(">> Modalità: CLASSIFICAZIONE")

    # 2. Cerchiamo tutte le parti (part_1, part_2, ecc.) di questo model_id
    print(f"Ricerca modelli su s3://{bucket_name}/{prefix} ...")
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if 'Contents' not in response:
            raise Exception(f"Nessun modello trovato per l'ID: {model_id}")
            
        # 3. Scarichiamo e carichiamo in RAM ogni pezzo
        models_dir = "/tmp/inference_models"
        os.makedirs(models_dir, exist_ok=True)
        
        for obj in response['Contents']:
            s3_key = obj['Key']
            if s3_key.endswith('.joblib'):
                filename = os.path.basename(s3_key)
                local_path = os.path.join(models_dir, filename)
                
                print(f" -> Scaricamento: {filename}")
                s3_client.download_file(bucket_name, s3_key, local_path)
                
                print(f" -> Caricamento in RAM di {filename}")
                clf = joblib.load(local_path)
                loaded_subforests.append(clf)
                
                # Salviamo il numero di feature attese usando il primo modello
                if model_feature_count == 0:
                    model_feature_count = clf.n_features_in_
                    
                # Pulizia disco
                os.remove(local_path)
                
        print(f"\n Modello {model_id} caricato con successo! Sotto-foreste attive: {len(loaded_subforests)}")
        print(f" Feature in ingresso attese: {model_feature_count}")
        
    except Exception as e:
        print(f" Errore critico durante il caricamento del modello: {e}")
        sys.exit(1)


# Endpoint REST per effettuare un'inferenza al volo
@app.route('/predict', methods=['POST'])
def predict_single_instance():
    
    # 1. Controlliamo che il modello sia caricato
    if not loaded_subforests or not current_strategy:
        return jsonify({"error": "Nessun modello caricato nel server"}), 500
        
    try:
        # 2. Riceviamo i dati JSON dal client
        data = request.get_json()
        
        if "features" not in data:
            return jsonify({"error": "Il JSON deve contenere la chiave 'features' con una lista di valori"}), 400
            
        features_list = data["features"]
        
        # Validazione della lunghezza
        if len(features_list) != model_feature_count:
            return jsonify({
                "error": f"Dimensione feature errata. Attese: {model_feature_count}, Ricevute: {len(features_list)}"
            }), 400

        # 3. Preparazione dati per Scikit-Learn
        # Il modello si aspetta una matrice 2D (1 riga, N colonne)
        input_matrix = np.array(features_list, dtype=np.float32).reshape(1, -1)
        
        # 4. Inferenza Parallela Locale (Interroghiamo ogni sotto-foresta)
        all_votes = []
        for clf in loaded_subforests:
            # clf.predict ritorna un array, prendiamo il primo elemento [0] perché c'è solo 1 riga
            vote = clf.predict(input_matrix)[0]
            all_votes.append(vote)
            
        # 5. Aggregazione usando le TUE classi strategiche
        final_prediction = current_strategy.aggregate(all_votes)
        
        # Convertiamo NumPy types in tipi Python nativi per la risposta JSON
        if isinstance(final_prediction, np.generic):
            final_prediction = final_prediction.item()
            
        # 6. Risposta
        return jsonify({
            "status": "success",
            "votes_from_subforests": [float(v) for v in all_votes], # Mostriamo anche i voti parziali per debug!
            "final_prediction": final_prediction
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Avvia il Server di Inferenza REST")
    parser.add_argument('--model_id', type=str, required=True, help="Es: rf_higgs_20260224_120956")
    parser.add_argument('--dataset', type=str, required=True, choices=['higgs', 'taxi', 'ids'], help="Il tipo di dataset per decidere la strategia")
    parser.add_argument('--port', type=int, default=8080, help="Porta per l'API (default 8080)")
    
    args = parser.parse_args()
    
    # Prima di avviare il server web, carichiamo la Foresta pesante in memoria!
    print(f"Avvio server di inferenza per il modello: {args.model_id}...")
    download_and_load_model(args.model_id, args.dataset)
    
    # Avvia l'API
    print(f"\n Server API pronto all'ascolto sulla porta {args.port}!")
    app.run(host='0.0.0.0', port=args.port)
