import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.core.strategies.strategies import TaskStrategy
from src.network.proto import rf_service_pb2


class RegressionStrategy(TaskStrategy):
    def get_task_type(self): return 1
    def extract_predictions(self, response): return response.estimates
    def aggregate(self, votes): return np.mean(votes) if votes else None
    def report(self, y_true, y_pred):

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)

        print("-" * 30)
        print(f"MSE (Mean Squared Error): {mse:.4f}")
        print(f"RMSE (Root Mean Sq. Error): {rmse:.4f}")
        print(f"MAE (Mean Absolute Error): {mae:.4f}")
        print("-" * 30)
        return {'mse': mse, 'rmse': rmse, 'mae': mae}

    def create_predict_response(self, results):
        # Impacchetta i float nel campo 'estimates'
        return rf_service_pb2.PredictResponse(estimates=results)