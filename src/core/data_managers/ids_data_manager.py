import os

import pandas as pd

from shard_dataset import shard_dataset, PATHS
from src.core.data_managers.data_managers import BaseDataManager


class IDSDataManager(BaseDataManager):
    def prepare_data(self, config):
        print(">>> TASK: CLASSIFICAZIONE (IDS Cybersecurity)")
        root_dir = config['_root_dir']
        rel_path = "data/ids/processed/ids_optimized.csv"
        full_path = os.path.join(root_dir, rel_path)

        df = pd.read_csv(full_path)
        # Casting Label intera
        if 'Label' in df.columns:
            df['Label'] = df['Label'].astype(int)

        # Chiamata alla logica comune con stratificazione
        train_df, test_df =  self._split_and_save_temp(df, config, stratify=df['Label'])

        NUM_WORKERS = config['num_workers']
        shard_dataset("ids", PATHS["ids"], NUM_WORKERS, self.strategy.get_task_type())
        return train_df, test_df

    def get_target_column(self):
        return 'Label'  #

    def get_shards_path(self, config):
        return config['paths']['ids_shards']