import json
import os


# Carica la configurazione risalendo alla root del progetto.
def load_config():
    # Trova la directory corrente (src/utils1)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Risale alla root (distributed_rf)
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))

    config_path = os.path.join(root_dir, 'config', 'config.json')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config non trovato in: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Aggiunge percorsi assoluti utili al sistema
    config['_root_dir'] = root_dir
    config['_data_dir'] = os.path.join(root_dir, 'data')
    config['_models_dir'] = os.path.join(root_dir, 'models')

    return config