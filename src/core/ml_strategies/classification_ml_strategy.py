import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.core.ml_strategies.ml_strategy import MLStrategy


class ClassificationMLStrategy(MLStrategy):
    def create_model(self, **kwargs):
        return RandomForestClassifier(**kwargs)

    def cast_target(self, y):
        return y.astype(np.int32)

    def format_tree_preds(self, model, input_matrix):
        # Forza int32 per i voti della classificazione
        preds_list = [tree.predict(input_matrix).astype(np.int32) for tree in model.estimators_]
        preds_matrix = np.vstack(preds_list)
        return preds_matrix.T.flatten().tolist()
