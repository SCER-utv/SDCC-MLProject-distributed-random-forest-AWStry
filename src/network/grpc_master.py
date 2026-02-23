from concurrent.futures import ThreadPoolExecutor, as_completed

import grpc
import numpy as np
import boto3   # [NUOVO] Per Auto-Healing
import time    # [NUOVO] Per le pause
from src.network.proto import rf_service_pb2_grpc, rf_service_pb2
import socket
import threading # Aggiungi questo import in alto

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
        
        # --- AGGIUNGI QUESTE DUE RIGHE QUI ---
        self.recovery_lock = threading.Lock()
        self.is_recovering = {}
        # -------------------------------------

    def _spawn_new_worker(self, old_worker_address):
        # 1. CONTROLLO VELOCE SENZA LOCK (per evitare colli di bottiglia se la macchina è già pronta)
        if old_worker_address in self.is_recovering and self.is_recovering[old_worker_address] is not None:
             print(f" ⏳ Attesa ripristino già completato per {old_worker_address}...")
             return self.is_recovering[old_worker_address]

        # 2. SEZIONE CRITICA CON LOCK
        with self.recovery_lock:
            # Doppia verifica (Double-checked locking) in caso un altro thread l'abbia creata mentre aspettavamo
            if old_worker_address in self.is_recovering:
                return self.is_recovering[old_worker_address]

            # Segniamo subito che abbiamo preso in carico il ripristino
            self.is_recovering[old_worker_address] = None

            # --- CALCOLO DEL NOME LOGICO ---
            try:
                worker_num = self.workers.index(old_worker_address) + 1
                new_name = f"DRF-worker{worker_num}-autohealed"
            except ValueError:
                new_name = "worker_extra_autohealed"
                worker_num = "?"
            # ---------------------------------------
            
            print(f"\n [AUTO-HEALING] Crash del nodo {old_worker_address}! Innesco ripristino unico (Lock acquisito)...")
            ec2 = boto3.resource('ec2', region_name='us-east-1')
            
            AMI_ID = 'ami-0314835b37682ec20'  
            SUBNET_ID = 'subnet-0a61f2346de4cd937'
            SG_ID = 'sg-004dedd411b0fe130'     
            KEY_NAME = 'distributed-random-forest-key'

            startup_script = """#!/bin/bash
            sudo -u ubuntu bash -c '
            cd /home/ubuntu/SDCC-MLProject-distributed-random-forest-AWStry
            source venv/bin/activate
            export PYTHONPATH=$(pwd)
            export AWS_S3_BUCKET=distributed-random-forest-bkt
            nohup python src/worker.py 50051 > /home/ubuntu/worker_log.txt 2>&1 &
            '
            """

            try:
                instances = ec2.create_instances(
                    ImageId=AMI_ID,
                    InstanceType='t3.large',
                    SubnetId=SUBNET_ID,
                    SecurityGroupIds=[SG_ID], 
                    KeyName=KEY_NAME,
                    UserData=startup_script,
                    MinCount=1, MaxCount=1,
                    IamInstanceProfile={
                        'Name': 'LabInstanceProfile' 
                    }
                )
                
                new_instance = instances[0]
                
                # <- [MODIFICA]: Aggiunto qui la forzatura del Tag
                new_instance.create_tags(
                    Tags=[
                        {
                            'Key': 'Name',
                            'Value': new_name
                        }
                    ]
                )

                print(f" [AUTO-HEALING] Creazione EC2 {new_instance.id} in corso. Attesa boot...")
                new_instance.wait_until_running()
                new_instance.reload()
                new_ip = new_instance.private_ip_address
                new_address = f"{new_ip}:50051"

                # --- ACTIVE POLLING ---
                port_is_open = False
                max_attempts = 30 # Prova per circa 2.5 minuti (30 * 5 sec)
                
                print(f" [AUTO-HEALING] Macchina accesa ({new_ip}). Attendo l'avvio di gRPC...")

                for attempt in range(max_attempts):
                    try:
                        with socket.create_connection((new_ip, 50051), timeout=2):
                            port_is_open = True
                            print(f" [AUTO-HEALING] Porta 50051 APERTA al tentativo {attempt + 1}! Il Worker è pronto.")
                            break
                    except (socket.timeout, ConnectionRefusedError, OSError):
                        print(f" Tentativo {attempt + 1}/{max_attempts}: Porta ancora chiusa. Attendo...")
                        time.sleep(5)
                
                if not port_is_open:
                    print(f" [AUTO-HEALING] Timeout critico: Il worker {new_ip} non ha aperto la porta in tempo utile.")
                    return None
                
                time.sleep(2) 
                
                # 3. SALVATAGGIO DELL'INDIRIZZO PER GLI ALTRI THREAD
                self.is_recovering[old_worker_address] = new_address
                return new_address

            except Exception as e:
                print(f" [AUTO-HEALING] Fallimento critico di Boto3: {e}")
                # Rimuoviamo il marcatore in caso di fallimento in modo che altri possano riprovare
                if old_worker_address in self.is_recovering:
                     del self.is_recovering[old_worker_address]
                return None

    def connect(self):
        print("--- Connessione ai Worker ---")

        # [MODIFICA FT] Connessione robusta: se uno è spento all'avvio, lo ignoriamo subito
        # ridondante
        active_stubs = []
        active_workers = [] # [NUOVO]

        for addr in self.workers:
            try:
                ch = grpc.insecure_channel(addr)
                grpc.channel_ready_future(ch).result(timeout=2)
                stub = rf_service_pb2_grpc.RandomForestWorkerStub(ch)
                self.channels.append(ch)
                active_stubs.append(stub)
                active_workers.append(addr) # [NUOVO]
                self.stubs.append(stub)
                print(f"Connesso a {addr}")
            except Exception as e:
                print(f"Errore connessione {addr}: {e}")

         # [NUOVO] Mantiene allineati indirizzi e stubs!
        self.stubs = active_stubs
        self.workers = active_workers
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
                'worker_addr': self.workers[i], # [NUOVO] Passiamo l'IP al thread
                'stub': self.stubs[i],
                'subforest_id': sub_id,
                'seed': i * 1000,
                'n_estimators': n_estimators,
                'dataset_path': worker_specific_path,
                'max_depth': conf['max_depth'],      
                'max_features': str(conf['max_features']), 
                'criterion': conf['criterion']
            })
            self.worker_assignments[sub_id] = (self.stubs[i], self.workers[i])

        def _execute_train_request(task):

            MAX_RETRIES = 1 
            current_stub = task['stub']
            current_addr = task['worker_addr']
            
            for attempt in range(MAX_RETRIES + 1):
                try:
                    req = rf_service_pb2.TrainRequest(
                        model_id=self.config['model_id'],
                        subforest_id=task['subforest_id'],
                        dataset_s3_path=task["dataset_path"], 
                        seed=task['seed'],
                        n_estimators=task['n_estimators'],
                        task_type=task_bit,  
                        max_depth=int(task['max_depth']),
                        max_features=str(task['max_features']),
                        criterion=str(task['criterion'])
                    )

                    print(f"Worker {task['subforest_id']} [{current_addr}] -> Inizio {task['n_estimators']} alberi...")
                    resp = current_stub.TrainSubForest(req, timeout=1200)
                    return task['subforest_id'], resp.success

                except grpc.RpcError as e:
                    print(f"\nCRASH RILEVATO: Worker {task['subforest_id']} ({current_addr}) è caduto durante il training!")
                    
                    if attempt < MAX_RETRIES:
                        print("🛠️ Innesco protocollo di ripristino per l'inferenza...")
                        new_addr_string = self._spawn_new_worker(current_addr)
                        
                        if new_addr_string:
                            # --- IL FIX È QUI: Ricreiamo il canale e lo Stub per il nuovo IP ---
                            ch = grpc.insecure_channel(new_addr_string)
                            new_stub = rf_service_pb2_grpc.RandomForestWorkerStub(ch)
                            
                            current_stub = new_stub
                            current_addr = new_addr_string
                            
                            # Aggiorniamo la rubrica globale (Thread-safe)
                            self.worker_assignments[task['subforest_id']] = (current_stub, current_addr)
                            # ------------------------------------------------------------------
                            
                            print(f" 🔄 Ritento l'inferenza del {sub_id} sul nuovo nodo {new_addr_string}...")
                            continue # Riavvia il ciclo per fare il secondo tentativo
                        else:
                            return sub_id, None
                    else:
                        print(f" Worker {task['subforest_id']} perso definitivamente. Rinuncio.")
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
                    print(f"Task {sid} FALLITO DEFINITIVAMENTE. Escludo worker.")
                    if sid in self.worker_assignments:
                        del self.worker_assignments[sid]
                    

    def predict_batch(self, batch_rows):
        # 1. Preparazione payload
        flat_feats = [x for r in batch_rows for x in r]
        row_votes = [[] for _ in range(len(batch_rows))]
        task_bit = self.strategy.get_task_type()

        # [MODIFICA] Inizializziamo un set per i nomi dei worker
        responded_ids = set()

    # 2. Richiesta parallela ai Worker con AUTO-HEALING
        def _ask_worker(sub_id, initial_stub, initial_addr):
            MAX_RETRIES = 1
            current_stub = initial_stub
            current_addr = initial_addr

            for attempt in range(MAX_RETRIES + 1):
                try:
                    req = rf_service_pb2.PredictRequest(
                        model_id=self.config['model_id'],
                        subforest_id=sub_id,
                        features=flat_feats,
                        task_type=task_bit
                        # NOTA: Qui il Worker deve essere programmato per sapere
                        # che se non ha il modello in RAM, deve prima scaricarlo da S3
                        # usando model_id e subforest_id come chiave!
                    )
                    
                    # Aumentiamo un po' il timeout perché, in caso di crash, 
                    # il nuovo worker dovrà scaricare il modello da S3 prima di rispondere
                    return sub_id, current_stub.Predict(req, timeout=30)

                except grpc.RpcError as e:
                    print(f"\n [AUTO-HEALING] CRASH INFERENZA: Worker {sub_id} ({current_addr}) non risponde!")
                    
                    if attempt < MAX_RETRIES:
                        print("🛠️ Innesco protocollo di ripristino per l'inferenza...")
                        new_addr = self._spawn_new_worker(current_addr)
                        
                        if new_addr:
                            ch = grpc.insecure_channel(new_addr)
                            current_stub = rf_service_pb2_grpc.RandomForestWorkerStub(ch)
                            current_addr = new_addr
                            
                            # Aggiorniamo la rubrica globale (Salvando la TUPLA!)
                            self.worker_assignments[sub_id] = (current_stub, current_addr)
                            
                            print(f" Ritento l'inferenza del {sub_id} sul nuovo nodo {new_addr}...")
                            continue # Riavvia il ciclo per fare il secondo tentativo
                        else:
                            return sub_id, None
                    else:
                        print(f" Worker {sub_id} perso definitivamente anche in inferenza.")
                        return sub_id, None
                        
                except Exception as e:
                    print(f"Errore sconosciuto su {sub_id}: {e}")
                    return sub_id, None

        # --- REINSERISCI QUESTO BLOCCO ALLA FINE DI PREDICT_BATCH ---
        active_workers = len(self.worker_assignments)
        if active_workers == 0:
            return [None] * len(batch_rows)

        with ThreadPoolExecutor(max_workers=active_workers) as executor:
            # Estraiamo direttamente la tupla (stub, addr) dal dizionario
            futures = {
                executor.submit(_ask_worker, sid, stub, addr): sid
                for sid, (stub, addr) in self.worker_assignments.items()
            }

            for f in as_completed(futures):
                sid, resp = f.result()
                if not resp: continue

                responded_ids.add(sid)
                worker_vals = self.strategy.extract_predictions(resp)
                if not worker_vals: continue

                n_trees_in_worker = len(worker_vals) // len(batch_rows)
                if n_trees_in_worker == 0: continue

                for i in range(len(batch_rows)):
                    start = i * n_trees_in_worker
                    end = start + n_trees_in_worker
                    row_votes[i].extend(worker_vals[start:end])

        # Aggregazione finale delegata alla strategia
        return [self.strategy.aggregate(vals) for vals in row_votes]
