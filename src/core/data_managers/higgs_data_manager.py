import os
# [MODIFICA] Rimossi gli import di pandas e numpy, non servono più qui!
from shard_dataset import shard_dataset, PATHS
from src.core.data_managers.data_managers import BaseDataManager

class HiggsDataManager(BaseDataManager):

    def prepare_data(self, config):
        print(">>> TASK: CLASSIFICAZIONE (Higgs dataset)")
        
        # [MODIFICA 1: RIMOZIONE CARICAMENTO RAM]
        # Abbiamo rimosso completamente il pd.read_csv() e il casting manuale in RAM.
        # Abbiamo rimosso la chiamata a self._split_and_save_temp().
        
        # [MODIFICA 2: INIEZIONE RIGIDA DEI WORKER]
        # Prende il valore rigorosamente dalla scatola config (iniettato dal Master)
        NUM_WORKERS = config['num_workers']
        
        # [MODIFICA 3: DELEGA ALLO SHARDER]
        # Chiamiamo lo sharder passando strategy=False (indica Classificazione/Stratificazione)
        shard_dataset("higgs", PATHS["higgs"], NUM_WORKERS, strategy=False)
        
        # [MODIFICA 4: RITORNA IL PERCORSO, NON IL DATAFRAME GIGANTE]
        # Il master si aspetta di sapere dove si trova il Test Set per calcolare l'Accuracy
        root_dir = config['_root_dir']
        test_path = os.path.join(root_dir, "data", "temp", "test_set.csv")
        return test_path

    def get_target_column(self):
        return 'Label'  

    def get_shards_path(self, config):
        # [MODIFICA 5: AGGIORNAMENTO PATH JSON]
        # Adattato alla nuova struttura annidata del file config.json
        return config['paths']['higgs']['shards_prefix']