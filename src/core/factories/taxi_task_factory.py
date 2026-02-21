from src.core.data_managers.taxi_data_manager import TaxiDataManager
from src.core.factories.factories import TaskFactory
from src.core.ml_strategies.regression_ml_strategy import RegressionMLStrategy
from src.core.strategies.regression_strategy import RegressionStrategy


class TaxiTaskFactory(TaskFactory):
    def create_strategy(self): return RegressionStrategy()
    def create_data_manager(self, strategy): return TaxiDataManager(strategy)
    def create_ml_strategy(self): return RegressionMLStrategy()