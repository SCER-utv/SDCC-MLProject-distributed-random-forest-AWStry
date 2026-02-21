from abc import ABC, abstractmethod

class TaskFactory(ABC):
    @abstractmethod
    def create_strategy(self): pass
    @abstractmethod
    def create_data_manager(self, strategy): pass
    @abstractmethod
    def create_ml_strategy(self): pass  # Nuovo metodo per il Worker