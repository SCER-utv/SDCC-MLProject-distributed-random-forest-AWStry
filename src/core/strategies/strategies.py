from abc import ABC, abstractmethod
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score

class TaskStrategy(ABC):
    @abstractmethod
    def extract_predictions(self, response): pass # Legge da Protobuf
    @abstractmethod
    def aggregate(self, votes): pass # Media o Moda
    @abstractmethod
    def report(self, y_true, y_pred): pass # Stampa metriche
    @abstractmethod
    def get_task_type(self): pass # Ritorna 0 o 1
    @abstractmethod
    def create_predict_response(self, results): pass