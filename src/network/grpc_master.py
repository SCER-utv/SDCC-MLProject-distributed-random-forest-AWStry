from concurrent.futures import ThreadPoolExecutor, as_completed

import grpc
import numpy as np

from src.network.proto import rf_service_pb2_grpc, rf_service_pb2

class GrpcMaster:
    def __init__(self, config, strategy):
        
        self.config = config
        self.strategy = strategy
        self.workers = config['workers']
        self.channels = []
        self.stubs = []
        self.worker_assignments = {}
        # [MODIFICA FT] Lista dei worker "morti" da escludere
        self.dead_workers = set()

    def connect(self):
        print("--- Connessione ai Worker ---")

        # [MODIFICA FT] Connessione robusta: se uno è spento all'avvio, lo ignoriamo subito
        # ridondante
        active_stubs = []

        for addr in self.workers:
            try:
                ch = grpc.insecure_channel(addr)
                grpc.channel_ready_future(ch).result(timeout=2)
                stub = rf_service_pb2_grpc.RandomForestWorkerStub(ch)
                self.channels.append(ch)
                #ridondante
                active_stubs.append(stub)
                self.stubs.append(stub)
                print(f"Connesso a {addr}")
            except Exception as e:
                print(f"Errore connessione {addr}: {e}")

        # [MODIFICA FT] Verifichiamo la presenza di worker attivi
        #ridondante
        self.stubs = active_stubs
        if not self.stubs:
            raise Exception("CRITICO: Nessun worker disponibile! Impossibile avviare il cluster.")


    def close(self):
        for ch in self.channels: ch.close()

    def train(self):
        # [POLIMORFISMO] Recuperiamo il bit del task dalla strategia
        task_bit = self.strategy.get_task_type()
        print(f"\n--- Avvio Training Distribuito ({self.strategy.__class__.__name__}) ---")

        # [MODIFICA ETEROGENEA] 1. DEFINIZIONE STRATEGIE
        # Creiamo profili diversi per i worker

        total_trees = self.config['total_trees']
        num_workers = len(self.stubs)

        if num_workers == 0:
            print("Nessun worker disponibile.")
            return
        
        trees_per_worker = total_trees // num_workers
        remainder = total_trees % num_workers

        tasks = []
        # Recuperiamo il template del path dal config (es: "data/taxi/shards/train_part_{}.csv")
        base_path_template = self.config.get("dataset_path")

        # [MODIFICA] Estraiamo la lista delle strategie passate dal master.py
        strategies = self.config.get('worker_strategies', [])

        for i in range(num_workers):
            n_estimators = trees_per_worker + (remainder if i == num_workers - 1 else 0)
            sub_id = f"part_{i + 1}"

            # [MODIFICA CRITICA] Peschiamo la singola configurazione dal JSON per questo specifico worker
            conf = strategies[i]

            # [LOGICA SHARDING]
            # Se il path contiene {}, lo formattiamo con l'ID del worker (i+1)
            # Altrimenti usiamo il path così com'è (fallback per test non shardati)
            if base_path_template and "{}" in base_path_template:
                worker_specific_path = base_path_template.format(i + 1)
            else:
                # Se non c'è la parentesi graffa, accodiamo il numero manualmente
                worker_specific_path = f"{base_path_template}{i + 1}.csv"

            # [MODIFICA] Prendiamo i parametri dinamici passati dal terminale!
            print(f" -> Assegnazione {sub_id} (Worker {i+1}): {worker_specific_path}")

            tasks.append({
                'stub': self.stubs[i],
                'subforest_id': sub_id,
                'seed': i * 1000,
                'n_estimators': n_estimators,
                'dataset_path': worker_specific_path,
                'max_depth': conf['max_depth'],      
                'max_features': str(conf['max_features']), 
                'criterion': conf['criterion']
            })
            self.worker_assignments[sub_id] = self.stubs[i]

        def _execute_train_request(task):
            try:
                req = rf_service_pb2.TrainRequest(
                    model_id=self.config['model_id'],
                    subforest_id=task['subforest_id'],
                    dataset_s3_path=task["dataset_path"], #fix per sharding
                    seed=task['seed'],
                    n_estimators=task['n_estimators'],
                    task_type=task_bit,  # Bit fornito dalla strategia

                    # [MODIFICA ETEROGENEA] 4. INIEZIONE NELLA RICHIESTA GRPC
                    max_depth=int(task['max_depth']),
                    max_features=str(task['max_features']),
                    criterion=str(task['criterion'])
                )

                print(f"Worker {task['subforest_id']} -> Inizio {task['n_estimators']} alberi...")
                resp = task['stub'].TrainSubForest(req, timeout=1200)
                return task['subforest_id'], resp.success

            except grpc.RpcError as e:
                # [MODIFICA FT] Gestione Crash durante Training
                print(f"CRASH RILEVATO: Worker {task['subforest_id']} è caduto durante il training!")
                return task['subforest_id'], False
            except Exception as e:
                print(f"Errore su {task['subforest_id']}: {e}")
                return task['subforest_id'], False

        completed_tasks = 0
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_execute_train_request, t) for t in tasks]
            for f in as_completed(futures):
                sid, success = f.result()
                if success:
                    print(f"Task {sid} completato.")
                    completed_tasks += 1
                else:
                    # [MODIFICA FT] Se fallisce, rimuoviamo il worker dalla lista degli attivi per l'inferenza
                    print(f"Task {sid} FALLITO. Escludo questo worker dal training e dall'inferenza")
                    if sid in self.worker_assignments:
                        del self.worker_assignments[sid]

    def predict_batch(self, batch_rows):
        # 1. Preparazione payload
        flat_feats = [x for r in batch_rows for x in r]
        row_votes = [[] for _ in range(len(batch_rows))]
        task_bit = self.strategy.get_task_type()

        # [MODIFICA] Inizializziamo un set per i nomi dei worker
        responded_ids = set()

        # 2. Richiesta parallela ai Worker
        def _ask_worker(sub_id, stub):
            try:
                req = rf_service_pb2.PredictRequest(
                    model_id=self.config['model_id'],
                    subforest_id=sub_id,
                    features=flat_feats,
                    task_type=task_bit
                )
                # [MODIFICA FT] Timeout stretto per l'inferenza (5 secondi)
                # Se un worker non risponde subito, lo consideriamo morto e non blocchiamo l'utente.
                return sub_id, stub.Predict(req, timeout=5)

            except grpc.RpcError:
                # [MODIFICA FT] Catturiamo silenziosamente l'errore
                # print(f"timeout/crash {sub_id}", end=" ") # Decommenta per debug
                return sub_id, None
            except Exception:
                return None

        # Usiamo solo i worker che hanno finito il training con successo (worker_assignments aggiornato)
        active_workers = len(self.worker_assignments)
        if active_workers == 0:
            return [None] * len(batch_rows)

        with ThreadPoolExecutor(max_workers=len(self.worker_assignments)) as executor:
            futures = {executor.submit(_ask_worker, sid, s): sid
                       for sid, s in self.worker_assignments.items()}

            for f in as_completed(futures):
                sid, resp = f.result()
                if not resp: continue

                # [MODIFICA] Aggiungiamo l'ID del worker alla lista dei rispondenti
                responded_ids.add(sid)

                # [POLIMORFISMO] Estrarre i valori corretti (votes vs estimates) tramite la strategia
                worker_vals = self.strategy.extract_predictions(resp)

                if not worker_vals: continue

                # Spacchettamento dei voti
                n_trees_in_worker = len(worker_vals) // len(batch_rows)
                if n_trees_in_worker == 0: continue

                for i in range(len(batch_rows)):
                    start = i * n_trees_in_worker
                    end = start + n_trees_in_worker
                    row_votes[i].extend(worker_vals[start:end])

        # [POLIMORFISMO] Aggregazione finale delegata alla strategia (Media o Moda)
        return [self.strategy.aggregate(vals) for vals in row_votes]