from abc import ABC, abstractmethod
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class MLStrategy(ABC):
    @abstractmethod
    def create_model(self, **kwargs): pass

    @abstractmethod
    def cast_target(self, y): pass

    @abstractmethod
    def format_tree_preds(self, preds_list): pass