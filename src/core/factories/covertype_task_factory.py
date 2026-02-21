from src.core.data_managers.covertype_data_manager import CovertypeDataManager
from src.core.data_managers.higgs_data_manager import HiggsDataManager
from src.core.factories.factories import TaskFactory
from src.core.ml_strategies.classification_ml_strategy import ClassificationMLStrategy
from src.core.strategies.classification_strategy import ClassificationStrategy


class CovertypeTaskFactory(TaskFactory):
    def create_strategy(self): return ClassificationStrategy()
    def create_data_manager(self, strategy): return CovertypeDataManager(strategy)
    def create_ml_strategy(self): return ClassificationMLStrategy()