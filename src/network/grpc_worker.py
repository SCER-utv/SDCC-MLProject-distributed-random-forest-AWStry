import os
import sys
import time
import traceback
from concurrent import futures

import grpc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.core.factories.ids_task_factory import IDSTaskFactory
from src.core.factories.taxi_task_factory import TaxiTaskFactory

from src.core.model import RandomForestManager
from src.network.proto import rf_service_pb2_grpc, rf_service_pb2


class GrpcWorker(rf_service_pb2_grpc.RandomForestWorkerServicer):
    def __init__(self, config):
        self.manager = RandomForestManager(config['_models_dir'])

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


