import os
import sys
from abc import abstractmethod, ABC
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from shard_dataset import shard_dataset, PATHS

class BaseDataManager(ABC):

    def __init__(self, strategy):
        # Salva la strategy come attributo dell'istanza
        self.strategy = strategy

    # Prepara i dati per il training, delegando il lavoro a shard_dataset
    @abstractmethod
    def prepare_data(self, config): pass

    @abstractmethod
    def get_target_column(self): pass

    # Recupera il prefisso degli shard dal nuovo JSON
    @abstractmethod
    def get_shards_path(self, config): pass