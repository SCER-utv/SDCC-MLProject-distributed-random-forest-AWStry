from src.core.data_managers.ids_data_manager import IDSDataManager
from src.core.factories.factories import TaskFactory
from src.core.ml_strategies.classification_ml_strategy import ClassificationMLStrategy
from src.core.strategies.classification_strategy import ClassificationStrategy


class IDSTaskFactory(TaskFactory):
    def create_strategy(self): return ClassificationStrategy()
    def create_data_manager(self, strategy): return IDSDataManager(strategy)
    def create_ml_strategy(self): return ClassificationMLStrategy()