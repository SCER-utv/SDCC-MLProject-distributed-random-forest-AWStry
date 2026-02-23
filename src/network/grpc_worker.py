import os
import sys
import time
import traceback
from concurrent import futures
import boto3
import grpc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.factories.ids_task_factory import IDSTaskFactory
from src.core.factories.taxi_task_factory import TaxiTaskFactory

from src.core.model import RandomForestManager
from src.network.proto import rf_service_pb2_grpc, rf_service_pb2


class GrpcWorker(rf_service_pb2_grpc.RandomForestWorkerServicer):
    def __init__(self, config):
        self.manager = RandomForestManager(config['_models_dir'])
        self.config = config
        self.models_dir = config['_models_dir']
        self.manager = RandomForestManager(self.models_dir)
        
        # [NUOVO] Setup di Boto3 per S3, HARDCODED, DA MODIFICARE
        self.s3_client = boto3.client('s3')
        self.bucket_name = os.getenv('AWS_S3_BUCKET', 'distributed-random-forest-bkt')

    def _get_factory(self, task_type):
        """Risoluzione dinamica della Factory"""
        return TaxiTaskFactory() if task_type == 1 else IDSTaskFactory()

    def HealthCheck(self, request, context):
        return rf_service_pb2.HealthStatus(alive=True)

    def TrainSubForest(self, request, context):
        try:

            print(f"[Worker] Ricevuta config: Depth={request.max_depth}, Feat={request.max_features}")

            # Il manager è già stato rifattorizzato per usare internamente le factory
            n = self.manager.train(
                model_id=request.model_id,
                subforest_id=request.subforest_id,
                dataset_path=request.dataset_s3_path,
                seed=request.seed,
                n_estimators=request.n_estimators,
                task_type=request.task_type,
                max_depth=request.max_depth,
                max_features=request.max_features,
                criterion=request.criterion
            )

            #  [NUOVO] SALVATAGGIO SU S3 
            # Assumiamo che il RandomForestManager salvi i file in formato .joblib.
            filename = f"{request.model_id}_{request.subforest_id}.joblib"
            local_model_path = os.path.join(self.models_dir, filename)
            s3_model_key = f"models_backup/{filename}"

            if os.path.exists(local_model_path):
                print(f"☁️ Upload modello su S3 in corso: {s3_model_key}...")
                self.s3_client.upload_file(local_model_path, self.bucket_name, s3_model_key)
                print(" Modello messo in sicurezza su S3.")
            else:
                print(f" Attenzione: Il file locale {local_model_path} non è stato trovato!")
           

            print(f"Sto usando il dataset: {request.dataset_s3_path}")
            return rf_service_pb2.TrainResponse(success=True, trees_built=n)
            
        except Exception:
            print(f"\n!!! ERRORE CRITICO TRAINING [{request.subforest_id}] !!!")
            traceback.print_exc()  # <--- Questo ti mostrerà l'errore esatto nel terminale worker
            return rf_service_pb2.TrainResponse(success=False)

    def Predict(self, request, context):
        try:

            #  [NUOVO] AUTO-HEALING: CONTROLLO E DOWNLOAD DA S3 
            filename = f"{request.model_id}_{request.subforest_id}.joblib"
            local_model_path = os.path.join(self.models_dir, filename)
            s3_model_key = f"models_backup/{filename}"

            # Se la macchina è appena "rinata", la cartella sarà vuota. Lo scarichiamo!
            if not os.path.exists(local_model_path):
                print(f"🚨 Modello {filename} non trovato in RAM/Disco!")
                print("🛠️ Innesco Auto-Healing: Download modello da S3...")
                
                # Creiamo la cartella models/checkpoints se la nuova EC2 non l'ha ancora creata
                os.makedirs(self.models_dir, exist_ok=True)
                
                # Scarichiamo il file
                self.s3_client.download_file(self.bucket_name, s3_model_key, local_model_path)
                print("✅ Modello ripristinato con successo da S3! Procedo all'inferenza.")
                
            # ---------------------------------------------------------
            # 1. Otteniamo i componenti polimorfici dalla Factory
            factory = self._get_factory(request.task_type)
            strategy = factory.create_strategy()

            # 2. Esecuzione dell'inferenza tramite il manager
            results = self.manager.predict_batch(
                model_id=request.model_id,
                subforest_id=request.subforest_id,
                flat_features=request.features,
                task_type=request.task_type
            )

            # 3. POLIMORFISMO: La strategia crea la risposta gRPC corretta
            # Sostituisce: if request.task_type == 1: ... else: ...
            return strategy.create_predict_response(results)

        except Exception as e:
            print(f"Errore durante l'Inferenza [{request.subforest_id}]: {e}")
            return rf_service_pb2.PredictResponse()


def run_server(port, config):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    rf_service_pb2_grpc.add_RandomForestWorkerServicer_to_server(GrpcWorker(config), server)
    server.add_insecure_port(f'[::]:{port}')
    server.start()

    print(f"Worker ONLINE: {port} (Premi CTRL+C per terminare)")

    try:
        while True:
            # Dorme per un giorno (o finché non viene interrotto)
            time.sleep(86400)
    except KeyboardInterrupt:

        # Ferma gRPC immediatamente
        server.stop(0)

        # Rilancia l'eccezione per farla gestire al worker.py
        raise


