import os
# [MODIFICA] Rimossi gli import di pandas e numpy
from shard_dataset import shard_dataset, PATHS
from src.core.data_managers.data_managers import BaseDataManager

class TaxiDataManager(BaseDataManager):
    
    def prepare_data(self, config):
        print(">>> TASK: REGRESSIONE (NYC Taxi)")
        
        # [MODIFICA 1: RIMOZIONE CARICAMENTO RAM]
        # Rimossi pd.read_csv, il ciclo for per il casting in float32 e il train_test_split in memoria.
        
        # [MODIFICA 2: INIEZIONE RIGIDA DEI WORKER]
        NUM_WORKERS = config['num_workers']
        
        # [MODIFICA 3: DELEGA ALLO SHARDER]
        # Chiamiamo lo sharder passando strategy=True (indica Regressione/KFold semplice)
        # shard_dataset("taxi", PATHS["taxi"], NUM_WORKERS, strategy=True)
        
        # [MODIFICA 4: RITORNA IL PERCORSO, NON IL DATAFRAME]
        # Il master userà questo file per calcolare il MSE (Mean Squared Error) alla fine
        root_dir = config['_root_dir']
        test_path = os.path.join(root_dir, "data", "temp", "test_set.csv")
        return test_path

    def get_target_column(self):
        return 'Label'  

    def get_shards_path(self, config):
        # [MODIFICA 5: AGGIORNAMENTO PATH JSON]
        # Adattato alla nuova struttura annidata del file config.json
        return config['paths']['taxi']['shards_prefix']