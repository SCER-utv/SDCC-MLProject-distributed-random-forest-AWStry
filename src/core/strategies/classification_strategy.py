import os
import sys

from src.core.strategies.strategies import TaskStrategy
from collections import Counter
# [MODIFICA 1] Aggiunti gli import per le nuove metriche
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.network.proto import rf_service_pb2


class ClassificationStrategy(TaskStrategy):
    def get_task_type(self): return 0
    def extract_predictions(self, response): return response.votes
    def aggregate(self, votes): return Counter(votes).most_common(1)[0][0] if votes else None
    def report(self, y_true, y_pred):
        # [MODIFICA 2] Calcolo delle nuove metriche (con zero_division=0 per evitare warning)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        print("-" * 30)
        print(f"ACCURACY FINALE: {accuracy:.2%}")
        print("-" * 30)

        # [MODIFICA] Restituiamo il dizionario arricchito per il salvataggio nel CSV!
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def create_predict_response(self, results):
        # Impacchetta gli int nel campo 'votes'
        return rf_service_pb2.PredictResponse(votes=results)