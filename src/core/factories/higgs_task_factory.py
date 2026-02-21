from src.core.data_managers.higgs_data_manager import HiggsDataManager
from src.core.factories.factories import TaskFactory
from src.core.ml_strategies.classification_ml_strategy import ClassificationMLStrategy
from src.core.strategies.classification_strategy import ClassificationStrategy


class HiggsTaskFactory(TaskFactory):
    def create_strategy(self): return ClassificationStrategy()
    def create_data_manager(self, strategy): return HiggsDataManager(strategy)
    def create_ml_strategy(self): return ClassificationMLStrategy()