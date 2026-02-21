import numpy as np
from sklearn.ensemble import RandomForestRegressor

from src.core.ml_strategies.ml_strategy import MLStrategy


class RegressionMLStrategy(MLStrategy):
    def create_model(self, **kwargs):
        return RandomForestRegressor(**kwargs)

    def cast_target(self, y):
        return y.astype(np.float32)

    def format_tree_preds(self, model, input_matrix):
        # Mantiene i float per i valori della regressione
        preds_list = [tree.predict(input_matrix) for tree in model.estimators_]
        preds_matrix = np.vstack(preds_list)
        return preds_matrix.T.flatten().tolist()