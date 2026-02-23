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
           
            print(f"Sto usando il dataset: {request.dataset_s3_path}")
            return rf_service_pb2.TrainResponse(success=True, trees_built=n)
            
        except Exception:
            print(f"\n!!! ERRORE CRITICO TRAINING [{request.subforest_id}] !!!")
            traceback.print_exc()  # <--- Questo ti mostrerà l'errore esatto nel terminale worker
            return rf_service_pb2.TrainResponse(success=False)

    def Predict(self, request, context):
        try:
            # 1. Otteniamo i componenti polimorfici dalla Factory
            factory = self._get_factory(request.task_type)
            strategy = factory.create_strategy()

            # --- RETRY LOGIC (Gestione S3 e ritardi di rete) ---
            max_retries = 3
            results = []
            success = False

            for attempt in range(max_retries):
                try:
                    # 2. Esecuzione dell'inferenza tramite il manager
                    # NOTA: Il manager gestisce in automatico il Lazy Loading da S3 se il file non è in RAM!
                    results = self.manager.predict_batch(
                        model_id=request.model_id,
                        subforest_id=request.subforest_id,
                        flat_features=request.features,
                        task_type=request.task_type
                    )
                    
                    # Se non ci sono eccezioni e ha restituito dei risultati, abbiamo vinto
                    if results is not None and len(results) > 0:
                        success = True
                        break
                    else:
                        print(f" Nessun risultato estratto. Ritento {attempt+1}/{max_retries}...")
                        time.sleep(2)

                except Exception as e:
                    error_msg = str(e)
                    if "404" in error_msg or "Not Found" in error_msg:
                        print(f" [S3 Eventual Consistency] Modello non ancora propagato. Ritento {attempt+1}/{max_retries}...")
                    else:
                        print(f" Errore imprevisto nell'inferenza: {e}. Ritento {attempt+1}/{max_retries}...")
                    
                    time.sleep(3) # Diamo tempo a S3 di allinearsi

            if not success:
                print(f" Fallimento critico: Impossibile completare l'inferenza per {request.subforest_id} dopo {max_retries} tentativi.")
                # Segnaliamo al Master che questo Worker ha fallito, così lui può fare l'Auto-Healing!
                context.abort(grpc.StatusCode.NOT_FOUND, "Modello non trovato su S3 o errore interno")
                return rf_service_pb2.PredictResponse()
            # ---------------------------------------------------

            # 3. POLIMORFISMO: La strategia crea la risposta gRPC corretta
            return strategy.create_predict_response(results)

        except Exception as e:
            print(f"Errore fatale durante l'Inferenza [{request.subforest_id}]: {e}")
            context.abort(grpc.StatusCode.INTERNAL, str(e))
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


