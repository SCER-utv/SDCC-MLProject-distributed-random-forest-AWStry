import sys
import os
import warnings
import time
import signal


from utils.config import load_config


# 2. Fix Path: Aggiunge la root del progetto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.network.grpc_worker import run_server

# 1. Pulizia dei Warning (Essenziale per non sporcare i log di training sui 7.3M di record)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
os.environ["PYTHONWARNINGS"] = "ignore"

# MODIFICA: Importiamo la funzione run_server dal nuovo file di rete


def signal_handler(sig, frame):
    """Gestisce la chiusura pulita del Worker."""
    print("\n\n[INFO] Ricevuto segnale di arresto. Chiusura Worker in corso...", end="")
    # Qui il server gRPC viene fermato dal KeyboardInterrupt catturato in run_server
    time.sleep(0.5)
    print(" [OK]")
    sys.exit(0)



if __name__ == '__main__':
    # Registra l'handler per CTRL+C
    signal.signal(signal.SIGINT, signal_handler)

    # Legge la porta dagli argomenti (default 50051)
    port = sys.argv[1] if len(sys.argv) > 1 else "50051"

    try:
        print(f"--- Inizializzazione Worker Node sulla porta {port} ---")
        config = load_config()

        # CONTROLLO: Assicuriamoci che nella config ci sia la cartella per i modelli
        if '_models_dir' not in config:
            config['_models_dir'] = "models/checkpoints"  # Fallback se manca nel file

        # Avvia il server gRPC (bloccante)
        # Passiamo la porta e la config al servizio che abbiamo scritto prima
        run_server(port, config)

    except KeyboardInterrupt:
        # Gestito via signal_handler
        pass
    except Exception as e:
        print(f"\n[ERRORE CRITICO] Avvio fallito: {e}")
        sys.exit(1)